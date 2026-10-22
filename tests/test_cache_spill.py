# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Disk eviction, memory pressure, and concurrent metadata access."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dynamix.engine.cache import Cache


def blob(n=100):
    return {'raster': np.arange(n, dtype=np.float32)}


def test_spill_load_and_reuse_disk_copy_without_computing(tmp_path):
    cache = Cache(max_bytes=1000, disk_dir=tmp_path/'cache', spill_threshold=1)
    cache.put('analysis|window', blob(), label='Leaders window')
    cache.put('b', blob())
    cache.wait_for_spills()
    assert 'analysis|window' not in cache._store
    assert cache.disk.record('analysis|window') is not None
    value = cache.get_or_compute('analysis|window', lambda: pytest.fail('disk hit computed'))
    np.testing.assert_array_equal(value['raster'], blob()['raster'])
    assert cache.disk.record('analysis|window') is not None
    assert all('|' not in p.name for p in cache.disk.folder.iterdir())
    assert any(e['event']=='hit on disk' for e in cache.snapshot()['events'])
    cache.close()
    assert not (tmp_path/'cache').exists()


def test_protected_entries_stay_and_small_ones_do_not_spill(tmp_path):
    c=Cache(max_bytes=1, disk_dir=tmp_path/'cache', spill_threshold=300)
    c.put('a',blob());c.pin('a')
    c.put('b',blob());c.mark_shown(1,'b')
    c.put('c',blob(),upstream='a')
    assert not c.spill('a') and not c.spill('b') and not c.spill('c')
    c.put('small',1)
    c.put('d',blob())
    c.wait_for_spills()
    assert c.disk.record('small') is None
    assert 'a' in c._store and 'b' in c._store
    c.close()


def test_large_victim_precedes_older_small_entries():
    c=Cache(max_bytes=5000,spill_threshold=1000)
    c.put('small',1)
    c.put('large',blob(500))
    c.put('new',blob(800))
    assert 'small' in c and 'large' not in c


def test_disk_budget_and_pickle_failure(tmp_path):
    c=Cache(max_bytes=1,disk_dir=tmp_path/'cache',disk_bytes=800,spill_threshold=1)
    for i in range(4): c.put(str(i),blob())
    c.wait_for_spills()
    assert c.snapshot()['disk']['bytes']<=800
    assert c.disk.record('0') is None
    c.put('unpicklable',lambda: None)
    c.put('last',blob())
    c.wait_for_spills()
    assert any(e['event']=='refusal to pickle' for e in c.snapshot()['events'])
    assert 'unpicklable' not in c
    c.close()


def test_pressure_releases_stages_and_evictable_results(tmp_path,monkeypatch):
    import psutil
    c=Cache(disk_dir=tmp_path/'cache',pressure_floor=100,spill_threshold=1)
    c.stages['stage']=np.zeros(3)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=0))
    c.put('a',blob());c.put('b',blob())
    c.wait_for_spills()
    assert len(c.stages)==0 and 'a' not in c._store and 'b' in c._store
    assert c.disk.record('a') is not None
    c.close()


def test_invalidation_during_write_cannot_resurrect(tmp_path,monkeypatch):
    c=Cache(max_bytes=1,disk_dir=tmp_path/'cache',spill_threshold=1)
    entered,release=threading.Event(),threading.Event()
    original=c.disk.write
    def slow(*args):
        entered.set(); assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(c.disk,'write',slow)
    c.put('a',blob());c.put('b',blob())
    assert entered.wait(5)
    assert c.invalidate('a')
    release.set();c.wait_for_spills()
    assert 'a' not in c and c.disk.record('a') is None
    c.close()


def test_snapshots_are_safe_during_insertions():
    c=Cache(maxsize=8)
    errors=[]
    def insert():
        try:
            for i in range(100):c.put(str(i),blob())
        except Exception as exc:errors.append(exc)
    worker=threading.Thread(target=insert);worker.start()
    while worker.is_alive():c.snapshot()
    worker.join()
    assert not errors and len(c)<=8


def test_automatic_budget_and_legacy_migration(tmp_path,monkeypatch):
    import psutil
    from dynamix.shell.settings import Settings,cache_ram_budget,load_settings,save_settings
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(total=32*1024**3))
    assert cache_ram_budget(Settings())==8*1024**3
    assert cache_ram_budget(Settings(cache_ram_bytes=123))==123
    path=tmp_path/'settings.json';monkeypatch.setenv('DYNAMIX_SETTINGS_PATH',str(path))
    for old,want in [(2_000_000_000,0),(345,345)]:
        path.write_text(json.dumps({'output_cache_bytes':old}),encoding='utf-8')
        settings=load_settings();assert settings.cache_ram_bytes==want
        save_settings(settings)
        assert 'output_cache_bytes' not in json.loads(path.read_text(encoding='utf-8'))


def test_scratch_sweep_preserves_live_process_directory(tmp_path,monkeypatch):
    import psutil
    from dynamix.shell.settings import scratch_dir,sweep_scratch
    monkeypatch.setenv('DYNAMIX_SETTINGS_PATH',str(tmp_path/'settings.json'))
    scratch=scratch_dir()
    for name in ['cache-123','cache-456']:
        d=scratch/name;d.mkdir();(d/'entry').write_text('data',encoding='utf-8')
    monkeypatch.setattr(psutil,'pid_exists',lambda pid:pid==123)
    sweep_scratch()
    assert (scratch/'cache-123'/'entry').exists()
    assert not (scratch/'cache-456').exists()


def test_concurrent_misses_share_one_compute():
    from concurrent.futures import ThreadPoolExecutor
    c=Cache()
    entered,release=threading.Event(),threading.Event()
    calls=[]
    def compute():
        calls.append(1);entered.set();assert release.wait(5)
        return blob()
    with ThreadPoolExecutor(2) as pool:
        first=pool.submit(c.get_or_compute,'a',compute)
        assert entered.wait(5)
        second=pool.submit(c.get_or_compute,'a',compute)
        release.set()
        assert first.result() is second.result()
    assert len(calls)==1


def test_invalidating_inflight_compute_prevents_reinsertion():
    c=Cache()
    def compute():
        c.invalidate('a')
        return blob()
    c.get_or_compute('a',compute)
    assert 'a' not in c


def test_disk_residency_does_not_take_the_gui_ram_hit_path(tmp_path):
    c=Cache(max_bytes=1,disk_dir=tmp_path/'cache',spill_threshold=1)
    c.put('a',blob());c.put('b',blob());c.wait_for_spills()
    assert 'a' not in c and c.disk.record('a') is not None
    assert c.get_or_compute('a',lambda:pytest.fail('disk entry recomputed')) is not None
    c.close()


def test_pressure_releases_multiple_small_entries(monkeypatch):
    import psutil
    c=Cache(pressure_floor=100)
    c.put('a',blob());c.put('b',blob());c.put('c',blob())
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=0))
    c.check_pressure()
    assert list(c._store)==['c']


def test_ram_reads_never_open_disk_files(tmp_path,monkeypatch):
    c=Cache(max_bytes=1,disk_dir=tmp_path/'cache',spill_threshold=1)
    c.put('a',blob());c.put('b',blob());c.wait_for_spills()
    def unexpected(*args):pytest.fail('RAM probe read a disk file')
    monkeypatch.setattr(c.disk,'read',unexpected)
    assert c.get('a') is None and not c.ready('a')
    c.close()


def test_ready_checks_the_resident_upstream_chain():
    c=Cache();c.put('a',blob());c.put('b',blob(),upstream='a')
    assert c.ready('b')
    c.invalidate('a')
    assert 'b' in c and not c.ready('b')
