# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The applicability rule over declared data contracts: a Transform that applies replaces the
available set with what it produces, a Filter leaves it, a step that does not apply leaves it and
says why; a device with no declarations always applies, and every step after an undeclared
Transform applies."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.pointset import PointSet
from dynamix.core.rasterfield import RasterField
from dynamix.model.chain import DeviceRef
from dynamix.model.contracts import (ALL_CONTRACTS, ALPHA, COMPONENTS, CONTRACT_KEYS, CONTRACTS,
                                     EXTREMA, FIELD, HMAP, POINTS, RECEIVED, STACK, TOPOLOGY,
                                     VCHAINS, StepState, consumes_of, input_contract,
                                     next_step_state, produces_of, producers_of, step_states)
from dynamix.model.device import register_device
from dynamix.model.param import Param, ParamKind


class _T:
    """A Transform shell: the rule never runs a device, it reads declarations."""

    params = ()

    def compute(self, field, params, *, progress=None):
        return {}

    def cache_key(self, source_id, params):
        return f"{self.name}:{source_id}"


class _F:
    params = ()

    def apply(self, result, params):
        return result


class ChainsSource(_T):
    name, label, families = "chains_src", "Chains source", ("Edge detection",)
    consumes = ((FIELD,),)
    produces = (EXTREMA, VCHAINS)


class ChainsSourceRoi(_T):
    name, label, families = "chains_src_roi", "Chains source, ROI", ("Edge detection",)
    consumes = ((FIELD,),)
    produces = (EXTREMA, VCHAINS)


class EdgesSource(_T):
    name, label, families = "edges_src", "Edges source", ("Edge detection",)
    consumes = ((FIELD,),)
    produces = (EXTREMA,)


class Graph(_T):
    name, label, families = "graph", "Graph", ("Topology",)
    consumes = ((EXTREMA, VCHAINS),)
    produces = (RECEIVED, TOPOLOGY)


class Stage(_T):
    name, label, families = "stage", "Stage", ("Data preparation",)
    consumes = ((FIELD,), (STACK,))
    produces = (RECEIVED,)


class Picker(_T):
    name, label, families = "picker", "Picker", ("Data preparation",)
    consumes = ((STACK,), (FIELD,))
    produces = (FIELD,)


class Bands(_T):
    name, label, families = "bands", "Bands", ("Decomposition",)
    consumes = ((STACK,),)
    produces = (COMPONENTS,)


class Either(_T):
    name, label, families = "either", "Either", ("Decomposition",)
    consumes = ((FIELD,), (STACK,))
    produces = (COMPONENTS,)


class Algo(_T):
    name, label, families = "algo", "Algo", ("Edge detection",)
    params = (Param("algorithm", ParamKind.CHOICE, default="with", choices=("with", "without")),)
    consumes = ((FIELD,),)

    @staticmethod
    def produces(params):
        return (EXTREMA, ALPHA) if params["algorithm"] == "with" else (EXTREMA,)


class Retired(_T):
    name, label, families = "retired", "Retired", ()
    consumes = ((FIELD,),)
    produces = (EXTREMA, VCHAINS)


class StubChains(_T):
    name = "stub_chains"
    consumes = ((FIELD,),)
    produces = (EXTREMA, VCHAINS)


class Loose(_T):
    name = "loose"


class ChainFilter(_F):
    name, label, families = "chain_f", "Chain filter", ("Selection: V-chains",)
    consumes = ((VCHAINS,),)
    produces = ()


class ExtremaFilter(_F):
    name, label, families = "extrema_f", "Extrema filter", ("Selection: extrema",)
    consumes = ((EXTREMA,),)
    produces = ()


class AlphaFilter(_F):
    name, label, families = "alpha_f", "Alpha filter", ("Selection: extrema",)
    consumes = ((EXTREMA, ALPHA),)
    produces = ()


class GraphFilter(_F):
    name, label, families = "graph_f", "Graph filter", ("Selection: V-chains",)
    consumes = ((TOPOLOGY,),)
    produces = ()


class Moded(_F):
    name, label, families = "moded", "Moded", ("Selection: extrema",)
    params = (Param("mode", ParamKind.CHOICE, default="here", choices=("here", "across")),)
    produces = ()

    @staticmethod
    def consumes(params):
        return ((EXTREMA, VCHAINS),) if params["mode"] == "across" else ((EXTREMA,),)


class LooseFilter(_F):
    name = "loose_f"


class ScalePicker(_F):
    """Stands in for the builtin scale_select: the level-view rule reads the device id."""

    name, label, families = "scale_select", "Scale", ("Selection: extrema",)
    consumes = ((EXTREMA,),)
    produces = ()


_ALL = (ChainsSource, ChainsSourceRoi, EdgesSource, Graph, Stage, Picker, Bands, Either, Algo,
        Retired, StubChains, Loose, ChainFilter, ExtremaFilter, AlphaFilter, GraphFilter, Moded,
        LooseFilter, ScalePicker)


@pytest.fixture
def registry(clean_registry):
    for cls in _ALL:
        register_device(cls())
    return clean_registry


def _steps(*names_or_pairs):
    out = []
    for item in names_or_pairs:
        name, params = (item, {}) if isinstance(item, str) else item
        out.append(DeviceRef(name, dict(params)))
    return tuple(out)


def _applies(steps, layer_input=FIELD):
    return [s.applies for s in step_states(steps, layer_input)]


def _raster(shape):
    ny, nx = shape[0], shape[1]
    return RasterField(name="r", values=np.zeros(shape), frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(nx)), y_axis=np.arange(float(ny)))


def test_the_nine_contracts_have_a_meaning_and_their_result_keys():
    assert set(CONTRACTS) == {FIELD, STACK, POINTS, EXTREMA, ALPHA, VCHAINS, TOPOLOGY, HMAP,
                              COMPONENTS}
    assert set(CONTRACT_KEYS) == set(CONTRACTS)
    assert CONTRACT_KEYS[EXTREMA] == ("extrema", "scales", "_shape")
    assert CONTRACT_KEYS[VCHAINS] == ("chains",)
    assert CONTRACT_KEYS[TOPOLOGY] == ("topology", "_topology_vids")
    assert CONTRACT_KEYS[HMAP] == ("h_map", "r2_map")
    assert CONTRACT_KEYS[COMPONENTS] == ("raster_out",)
    assert CONTRACT_KEYS[ALPHA] == CONTRACT_KEYS[FIELD] == CONTRACT_KEYS[STACK] == ()
    assert CONTRACT_KEYS[POINTS] == ()
    assert ALL_CONTRACTS == frozenset(CONTRACTS)
    assert RECEIVED not in ALL_CONTRACTS


def test_the_layer_input_is_a_field_a_stack_or_a_point_set():
    assert input_contract(_raster((8, 8))) == FIELD
    assert input_contract(_raster((8, 8, 3))) == STACK
    assert input_contract(_raster((8, 8, 1))) == FIELD
    assert input_contract(PointSet(lon=np.zeros(3), lat=np.zeros(3))) == POINTS


def test_a_bare_array_none_or_an_opaque_object_never_raises():
    assert input_contract(np.zeros((8, 8))) == FIELD
    assert input_contract(np.zeros((8, 8, 3))) == STACK
    assert input_contract(np.zeros((8, 8, 1))) == FIELD
    assert input_contract(None) == FIELD
    assert input_contract(object()) == FIELD


def test_a_transform_replaces_the_available_set_with_what_it_produces(registry):
    assert _applies(_steps("chains_src", "chain_f")) == [True, True]
    assert _applies(_steps("edges_src", "chain_f")) == [True, False]
    assert _applies(_steps("edges_src", "either")) == [True, False]


def test_a_filter_leaves_the_available_set_unchanged(registry):
    assert _applies(_steps("chains_src", "extrema_f", "chain_f")) == [True, True, True]


def test_a_step_that_does_not_apply_leaves_the_set_and_the_analysis(registry):
    steps = _steps("edges_src", "graph", "extrema_f", "graph_f")
    assert _applies(steps) == [True, False, True, False]
    assert step_states(steps, FIELD)[3].reason.endswith(
        "the current analysis (Edges source) does not")


def test_received_passes_through_an_input_preserving_step(registry):
    assert _applies(_steps("stage", "chains_src"), FIELD) == [True, True]
    assert _applies(_steps("stage", "bands"), STACK) == [True, True]
    assert _applies(_steps("stage", "chains_src"), STACK) == [True, False]
    assert _applies(_steps("chains_src", "graph", "chain_f", "graph_f")) == [True] * 4


def test_consumes_alternatives_are_any_of(registry):
    assert _applies(_steps("picker"), FIELD) == [True]
    assert _applies(_steps("picker"), STACK) == [True]
    assert _applies(_steps("picker"), POINTS) == [False]
    assert _applies(_steps("picker", "chains_src"), STACK) == [True, True]


def test_declarations_may_be_functions_of_the_params(registry):
    from dynamix.model.device import get_device

    algo, moded = get_device("algo"), get_device("moded")
    assert produces_of(algo, {"algorithm": "with"}, frozenset({FIELD})) == {EXTREMA, ALPHA}
    assert produces_of(algo, {"algorithm": "without"}, frozenset({FIELD})) == {EXTREMA}
    assert consumes_of(moded, {"mode": "across"}) == ((EXTREMA, VCHAINS),)
    assert _applies(_steps("algo", "alpha_f")) == [True, True]
    assert _applies(_steps(("algo", {"algorithm": "without"}), "alpha_f")) == [True, False]
    assert _applies(_steps("edges_src", "moded")) == [True, True]
    assert _applies(_steps("edges_src", ("moded", {"mode": "across"}))) == [True, False]
    assert _applies(_steps("chains_src", ("moded", {"mode": "across"}))) == [True, True]


def test_an_undeclared_device_always_applies_and_every_step_after_it_does(registry):
    from dynamix.model.device import get_device

    assert consumes_of(get_device("loose"), {}) == ()
    assert produces_of(get_device("loose"), {}, frozenset({FIELD})) == ALL_CONTRACTS
    assert _applies(_steps("loose", "loose_f"), POINTS) == [True, True]
    assert _applies(_steps("chains_src", "loose", "chain_f")) == [True, True, True]
    assert _applies(_steps("loose", "chain_f", "graph_f", "alpha_f")) == [True] * 4
    assert _applies(_steps("loose", "bands"), POINTS) == [True, True]
    assert _applies(_steps("loose", "edges_src", "chain_f")) == [True, True, False]


def test_the_reason_names_the_need_its_producers_and_the_current_analysis(registry):
    state = next_step_state(_steps("edges_src"), FIELD, "chain_f")
    assert state == StepState(False, "needs V-chains: Chains source and Chains source, ROI "
                                     "produce it; the current analysis (Edges source) does not")
    state = next_step_state(_steps("algo"), FIELD, "graph_f")
    assert state.reason == ("needs the H/V topology: Graph produces it; the current analysis "
                            "(Algo) does not")


def test_before_any_transform_the_reason_names_the_layer_input(registry):
    state = next_step_state((), FIELD, "bands")
    assert state == StepState(False, "needs a band stack; the layer's input is a single-band "
                                     "field")
    state = next_step_state((), POINTS, "either")
    assert state.reason == ("needs a single-band field or a band stack: Picker produces it; "
                            "the layer's input is a point set")


def test_next_step_state_places_a_transform_after_the_transforms(registry):
    steps = _steps("chains_src", "chain_f")
    assert next_step_state(steps, FIELD, "graph").applies
    assert not next_step_state(steps, FIELD, "graph_f").applies
    assert next_step_state(steps, FIELD, "moded", {"mode": "across"}).applies
    assert next_step_state((), STACK, "bands").applies


def test_under_a_shown_level_the_transport_sets_the_scale(registry):
    steps = _steps("edges_src", "scale_select", "extrema_f")
    assert step_states(steps, FIELD) == [StepState(True)] * 3
    assert step_states(steps, FIELD, level_view=True) == [
        StepState(True), StepState(False, "the scale is set by the Transport"), StepState(True)]
    assert next_step_state(_steps("edges_src"), FIELD, "scale_select").applies
    assert next_step_state(_steps("edges_src"), FIELD, "scale_select",
                           level_view=True) == StepState(False,
                                                         "the scale is set by the Transport")
    assert next_step_state((), FIELD, "scale_select", level_view=True).reason == (
        "the scale is set by the Transport")
    assert next_step_state(_steps("edges_src"), FIELD, "extrema_f", level_view=True).applies


def test_params_are_validated_inside_the_rule(registry):
    with pytest.raises(ValueError, match="unknown parameter"):
        step_states(_steps(("moded", {"bogus": 1})), FIELD)


def test_producers_leave_out_stubs_and_devices_in_no_family(registry):
    assert producers_of(VCHAINS) == ("chains_src", "chains_src_roi")
    assert producers_of(ALPHA) == ("algo",)
    assert producers_of(STACK) == ()


# --- the builtin devices ----------------------------------------------------------------------

@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def test_a_chain_filter_after_a_chainless_analyzer_says_who_makes_chains(builtins):
    state = next_step_state(_steps("cdf_edges"), FIELD, "chain_holder")
    assert state == StepState(False, "needs V-chains: WTMM 2D and WTMM 2D, ROI produce it; the "
                                     "current analysis (CDF edges) does not")
    state = next_step_state(_steps("cdf_edges"), FIELD, "hline_holder")
    assert state.reason == ("needs a per-point α: Mallat–Zhong edges produces it; the current "
                            "analysis (CDF edges) does not")
    assert not next_step_state(_steps("mz_edges", "chain_topology"), FIELD, "min_vchains").applies


def test_under_a_shown_level_the_builtin_scale_select_does_not_apply(builtins):
    steps = _steps("wtmm2d", "scale_select", "chain_holder")
    assert step_states(steps, FIELD) == [StepState(True)] * 3
    assert step_states(steps, FIELD, level_view=True) == [
        StepState(True), StepState(False, "the scale is set by the Transport"), StepState(True)]
