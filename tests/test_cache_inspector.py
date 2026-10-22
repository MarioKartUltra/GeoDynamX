# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Cache inspector ownership, actions and tool-window lifecycle."""
import numpy as np
from PySide6 import QtCore,QtWidgets

from dynamix.engine.cache import Cache
from dynamix.shell.cache_inspector import CacheInspector


def test_inspector_lists_tiers_and_holders_and_handles_actions(qtbot,tmp_path,monkeypatch):
    cache=Cache(disk_dir=tmp_path/'cache',spill_threshold=1)
    cache.put('a',{'raster':np.arange(100)},label='Leaders level 1')
    cache.put('pinned',{'raster':np.arange(100)},label='Frozen analysis');cache.pin('pinned')
    inspector=CacheInspector(cache,lambda snapshot:{'a':['Layer A']})
    qtbot.addWidget(inspector);inspector.show()
    assert inspector.windowFlags() & QtCore.Qt.WindowType.Tool
    assert inspector.timer.isActive()
    def select(key):
        inspector.refresh()
        index=next(i for i,row in enumerate(inspector._rows) if row['key']==key)
        inspector.table.clearSelection();inspector.table.selectRow(index)
    select('pinned');assert not inspector.drop.isEnabled()
    select('a');assert inspector.spill.isEnabled()
    inspector._spill();qtbot.waitUntil(lambda:not inspector._busy)
    cache.wait_for_spills();inspector.refresh()
    select('a');assert inspector.load.isEnabled()
    inspector._load();qtbot.waitUntil(lambda:not inspector._busy)
    assert 'a' in cache._store
    select('a');inspector._drop();qtbot.waitUntil(lambda:not inspector._busy)
    assert 'a' not in cache
    assert inspector.log.toPlainText()
    inspector.grab().save(str(tmp_path/'cache-inspector.png'))
    inspector.close();assert not inspector.timer.isActive()
    cache.close()


def test_shown_drop_requires_confirmation_and_next_request_computes(qtbot,monkeypatch):
    cache=Cache();cache.put('a',{'raster':np.arange(3)});cache.mark_shown(1,'a')
    inspector=CacheInspector(cache);qtbot.addWidget(inspector)
    inspector.table.selectRow(0)
    asked=[]
    monkeypatch.setattr(inspector,'_confirm',lambda text:asked.append(text) or False)
    inspector._drop();assert 'a' in cache and asked
    monkeypatch.setattr(inspector,'_confirm',lambda text:True)
    inspector._drop();qtbot.waitUntil(lambda:not inspector._busy)
    calls=[];cache.get_or_compute('a',lambda:calls.append(1) or {})
    assert calls==[1]


def test_main_window_command_opens_one_inspector(window,qtbot):
    assert getattr(window,'_cache_inspector',None) is None
    window.commands['view.cache_inspector'].trigger()
    first=window._cache_inspector
    assert first.isVisible()
    first.close();window.commands['view.cache_inspector'].trigger()
    assert window._cache_inspector is first and first.isVisible()


from tests.test_shell_window import stub_devices,window


def test_stages_appear_and_can_be_dropped(qtbot):
    cache=Cache();cache.stages['stage-key']={'cwt':np.arange(10.)}
    inspector=CacheInspector(cache);qtbot.addWidget(inspector)
    row=next(i for i,r in enumerate(inspector._rows) if r['kind']=='stage')
    inspector.table.selectRow(row)
    assert not inspector.spill.isEnabled()
    inspector._drop();qtbot.waitUntil(lambda:not inspector._busy)
    assert 'stage-key' not in cache.stages
    assert cache.snapshot()['events'][-1]['bytes']==80


def test_layer_provider_names_shown_freeze_and_mark_holders(window,monkeypatch):
    from types import SimpleNamespace
    layer=SimpleNamespace(layer_id=7,name='Region A',tags={'ui.freeze':'1'})
    window.project.layers=[layer]
    window._out_keys={7:{'output'}}
    window._marked_keys={(7,'leader',2):'output'}
    monkeypatch.setattr(window,'_cache_keys_for',lambda layer:['analysis'])
    monkeypatch.setattr(window,'_output_keys',lambda layer:{'leader':'output'})
    owners=window._cache_holders({'entries':[{'key':'analysis','holders':['shown']}],
                                  'shown':{7:'analysis'}})
    assert 'shown by Region A' in owners['analysis']
    assert 'Freeze in Region A' in owners['analysis']
    assert 'marked level in Region A' in owners['output']
    window.project.layers=[]
