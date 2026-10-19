# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Export records describe landed computations, including reused intermediate work."""
import json

import numpy as np
import pytest

from conftest import make_field
from dynamix.core import fft_policy
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.project import Project


@pytest.fixture
def setup(clean_registry, tmp_path):
    from dynamix.devices import register_builtin_devices
    register_builtin_devices()
    fft_policy.configure('fftw', 64)
    project = Project()
    source = project.add_source('input.npz', sha256='a'*64)
    layer = project.add_layer('test', source.source_id, Chain((DeviceRef('wtmm2d', {
        'n_oct': 1, 'n_voice': 2, 'precision': 'f64'}),)))
    out = tmp_path/'out.npz'
    out.write_bytes(b'exported bytes')
    return project, layer, make_field(shape=(32, 32)), out


def test_stale_recipe_refuses_without_replacing_a_sidecar(setup):
    from dynamix.engine.gprv_export import write_record
    project, layer, field, out = setup
    result = resolve(layer, field, Cache()).result
    sidecar = write_record(project, layer, out, export='chains_npz', result=result)
    original = sidecar.read_bytes()
    layer.chain = Chain((DeviceRef('wtmm2d', {'n_oct': 2, 'precision': 'f64'}),))
    with pytest.raises(ValueError, match='recipe|landed'):
        write_record(project, layer, out, export='chains_npz', result=result)
    assert sidecar.read_bytes() == original


def test_filter_change_is_detected_even_when_transform_key_matches(setup):
    from dynamix.engine.gprv_export import write_record
    project, layer, field, out = setup
    layer.chain = Chain(layer.chain.steps+(DeviceRef('chain_length', {'min_len': 2}),))
    result = resolve(layer, field, Cache()).result
    layer.chain = Chain(layer.chain.steps[:-1]+(DeviceRef('chain_length', {'min_len': 9}),))
    with pytest.raises(ValueError, match='recipe|landed'):
        write_record(project, layer, out, export='chains_npz', result=result)


def test_compute_context_survives_policy_change_and_cached_reuse(setup):
    from dynamix.engine.gprv_export import record_for
    project, layer, field, out = setup
    cache = Cache()
    first = resolve(layer, field, cache).result
    fft_policy.configure('fftw', 32)
    second = resolve(layer, field, cache).result
    a = record_for(project, layer, out, export='chains_npz', result=first)
    b = record_for(project, layer, out, export='chains_npz', result=second)
    for rec in (a, b):
        rec['processStep'][-1].pop('stepDateTime')
    assert a == b
    assert a['processStep'][0]['executionContext']['precision'] == 'float64'


def test_actual_fft_observer_distinguishes_mixed_precision_and_ignores_no_fft():
    fft_policy.configure('fftw', 32)
    with fft_policy.observing() as calls:
        fft_policy.active()  # selecting a backend is not a computation
    assert calls == []
    with fft_policy.observing() as calls:
        fft_policy.active().fft2(np.ones((4, 4)))
        fft_policy.double().fft2(np.ones((4, 4)))
    assert [c['precision'] for c in calls] == ['float32', 'float64']


def test_stage_cache_carries_represented_fft_context():
    from dynamix.engine.cache import StageStore
    store = StageStore()
    fft_policy.configure('fftw', 64)
    with fft_policy.observing() as fresh:
        store['cwt'] = fft_policy.double().fft2(np.ones((4, 4)))
    fft_policy.configure('fftw', 32)
    with fft_policy.observing() as reused:
        store['cwt']
    assert fresh == reused and reused[0]['precision'] == 'float64'


def test_field_stages_keep_their_own_resolved_outcomes(setup):
    from dynamix.engine.gprv_export import record_for
    project, layer, field, out = setup
    layer.chain = Chain((DeviceRef('fractional_integration', {'integration_order': .4}),
                         DeviceRef('interpolating_wavelet_filter', {'levels': 2, 'threshold': .1})))
    result = resolve(layer, field, Cache()).result
    record = record_for(project, layer, out, export='derived_npz', result=result)
    first, second = record['processStep'][:2]
    reports = lambda s: {r['name']: r['description'] for r in s.get('report', [])}
    assert reports(first)['method']['fractional_integration'][-1]['order'] == .4
    assert reports(second)['method']['preprocessing'][-1]['method'] == 'interpolating_wavelet_filter'
    assert first['executionContext']['backend']['name'] == 'fftw'
    assert 'backend' not in second['executionContext']  # lifting is not an FFT


def test_unreadable_parent_has_a_distinct_message(setup):
    from dynamix.engine.gprv_export import record_for
    project, layer, field, out = setup
    source = project.sources[layer.source_id]
    source.path = str(out)
    source.sha256 = __import__('hashlib').sha256(out.read_bytes()).hexdigest()
    out.with_name(out.name+'.gprv').write_text('{broken', encoding='utf-8')
    record = record_for(project, layer, out, export='derived_npz')
    assert 'unreadable' in record['source'][0]['description']
    assert 'other bytes' not in record['source'][0]['description']


def test_stage_reuse_does_not_change_scientific_record(setup):
    from dynamix.engine.gprv_export import record_for
    project, layer, field, out = setup
    cache = Cache()
    first = resolve(layer, field, cache).result
    # Repartition from a retained CWT, then restore the original moments.
    p = dict(layer.chain.steps[0].params)
    layer.chain = Chain((DeviceRef('wtmm2d', {**p, 'q_step': 1.}),))
    resolve(layer, field, cache)
    layer.chain = Chain((DeviceRef('wtmm2d', p),))
    # Evict only full results, retain intermediate stages.
    cache._store.clear()
    reused = resolve(layer, field, cache).result
    a = record_for(project, layer, out, export='chains_npz', result=first)
    b = record_for(project, layer, out, export='chains_npz', result=reused)
    for rec in (a, b):rec['processStep'][-1].pop('stepDateTime')
    assert a == b


def test_mixed_precision_integration_records_both_backends_without_false_step_precision(setup):
    from dynamix.engine.gprv_export import write_record
    project, layer, field, out = setup
    fft_policy.configure('fftw', 32)
    layer.chain = Chain((DeviceRef('wtmm2d', {'n_oct': 1, 'n_voice': 2, 'precision': 'f32',
                                             'integration_order': .4}),))
    result = resolve(layer, field, Cache()).result
    record = json.loads(write_record(project, layer, out, export='chains_npz', result=result).read_bytes())
    step = record['processStep'][0]
    assert 'precision' not in step['executionContext']
    params = step['processingInformation']['runTimeParameters']
    assert params['fft_fftw_float64']['precision'] == 'float64'
    assert params['fft_fftw_float32']['precision'] == 'float32'


@pytest.mark.parametrize('device,params', [
    ('dyadic_interpolation', {'levels': 1}),
    ('daf_smoothing', {'sigma': 2., 'order': 8}),
    ('interpolating_wavelet_filter', {'bank': 'gldaf', 'levels': 2, 'threshold': .1}),
    ('fractional_integration', {'integration_order': .4}),
])
def test_preparation_record_replays_the_actual_export_bytes(setup, device, params):
    from dynamix.engine.gprv_export import write_record, recipe_from_record
    project, layer, field, out = setup
    layer.chain = Chain((DeviceRef(device, params),))
    prepared = resolve(layer, field, Cache()).result
    prepared.save_npz(out)
    record = json.loads(write_record(project, layer, out, export='derived_npz', result=prepared).read_bytes())
    layer.chain = Chain(tuple(recipe_from_record(record)))
    replay = resolve(layer, field, Cache()).result
    replay_path = out.with_name('replay.npz')
    replay.save_npz(replay_path)
    assert out.read_bytes() == replay_path.read_bytes()


def test_code_edited_since_startup_refuses_to_mint_a_record(setup, monkeypatch):
    from dynamix import code_identity
    from dynamix.engine.gprv_export import write_record
    project, layer, field, out = setup
    result = resolve(layer, field, Cache()).result
    changed = {**code_identity.STARTUP, 'patch': b'edited source'}
    monkeypatch.setattr(code_identity, 'code_identity', lambda: changed)
    with pytest.raises(ValueError, match='source code changed'):
        write_record(project, layer, out, export='chains_npz', result=result)
    assert not out.with_name(out.name+'.gprv').exists()


def test_unknown_parent_version_is_cited_when_its_output_hash_matches(setup):
    from dynamix.engine.gprv_export import record_for
    import hashlib
    project, layer, field, out = setup
    source = project.sources[layer.source_id]
    source.path = str(out)
    source.sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
    data = json.dumps({'gprv': '0.4', 'processStep': [{'output': [{'sourceCitation': {
        'sha256': source.sha256}}]}]}).encode()
    out.with_name(out.name+'.gprv').write_bytes(data)
    record = record_for(project, layer, out, export='derived_npz')
    assert record['source'][0]['sourceMetadata'][0]['sha256'] == hashlib.sha256(data).hexdigest()


def test_lazy_output_identity_is_checked_before_recording(setup):
    from dynamix.engine.execution import attach, metadata
    from dynamix.engine.gprv_export import write_record
    from dynamix.engine.resolve import resolve_output
    project, layer, field, out = setup
    layer.chain = Chain((DeviceRef('interpolating_wavelets', {'levels': 2, 'show': 'horizontal'}),))
    cache = Cache()
    analysis = resolve(layer, field, cache).result
    output = resolve_output(layer, field, cache, 'horizontal', level=1)
    product = attach(analysis, _output_identity=output['_output_identity'],
                     _output_execution=metadata(output)['_execution'])
    write_record(project, layer, out, export='derived_npz', result=product,
                 print_params={'output': 'interpolating_wavelets/horizontal', 'level': 1})
    with pytest.raises(ValueError, match='landed'):
        write_record(project, layer, out, export='derived_npz', result=product, level=0)


def test_versioned_install_without_a_checkout_can_identify_its_release(monkeypatch):
    from dynamix import code_identity
    release = {'commit': None, 'version': '1.2.3', 'patch': None, 'checked': False}
    monkeypatch.setattr(code_identity, 'STARTUP', release)
    monkeypatch.setattr(code_identity, 'code_identity', lambda: dict(release))
    assert code_identity.checked_identity() == release
