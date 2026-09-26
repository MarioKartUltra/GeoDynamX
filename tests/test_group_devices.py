# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""``group_paint``/``group_filter`` -- the commit transaction's engine half.

``group_paint`` reads the committed-groups spec the WINDOW writes into its ``spec_json`` param at
commit time (the job) and stamps ``group:<name>`` tags onto COPIES of member chains, never
originals. ``group_filter`` keeps/drops chains by that tag -- an ordinary chain filter.
"""
import json

import pytest

from dynamix.devices.groups import GroupFilter, GroupPaint, decode_groups, encode_groups
from dynamix.model.param import Param, ParamKind
from dynamix.shell.knobs import control_spec


def _chain(idx=0):
    return {"id": idx, "x": [1, 2, 3]}


def _paint_params(**over):
    d = {p.name: p.default for p in GroupPaint.params}
    d.update(over)
    return d


def _filter_params(**over):
    d = {p.name: p.default for p in GroupFilter.params}
    d.update(over)
    return d


def _tagged_chain(tags):
    return {"tags": list(tags)}


# --- encode/decode ---------------------------------------------------------

def test_encode_decode_round_trip_including_color():
    groups = {"fault_a": {"signature": "sig1", "chains": [0, 2], "color": [255, 0, 0]}}
    text = encode_groups(groups)
    assert isinstance(text, str)
    assert decode_groups(text) == groups


def test_encode_decode_round_trip_multiple_groups():
    groups = {
        "fault_a": {"signature": "sigA", "chains": [0], "color": [255, 0, 0]},
        "fault_b": {"signature": "sigB", "chains": [1, 2], "color": [0, 255, 0]},
    }
    assert decode_groups(encode_groups(groups)) == groups


def test_decode_empty_string_is_empty_dict():
    assert decode_groups("") == {}


# --- ParamKind.TEXT ----------------------------------------------------

def test_paramkind_text_validate_passes_strings_through():
    p = Param("spec_json", ParamKind.TEXT, default="")
    assert p.validate("hello") == "hello"
    assert p.validate("") == ""


def test_paramkind_text_builds_control_spec_without_raising():
    p = Param("spec_json", ParamKind.TEXT, default="")
    spec = control_spec(p)
    assert spec.widget == "label"


def test_paramkind_text_rejects_range_declaration():
    with pytest.raises(ValueError):
        Param("spec_json", ParamKind.TEXT, default="", min=0.0, max=1.0)


def test_editable_text_param_builds_a_text_edit_control_spec():
    """``editable=True`` is the opt-in that earns a real editor -- default False
    (every existing TEXT param, including group_paint's own spec_json/signature below) stays a
    read-only "label", unaffected."""
    p = Param("group", ParamKind.TEXT, default="", editable=True)
    assert control_spec(p).widget == "text_edit"


# --- GroupPaint --------------------------------------------------------

def test_group_paint_empty_spec_json_is_passthrough_identity():
    result = {"chains": [_chain(0)]}
    out = GroupPaint().apply(result, _paint_params())
    assert out is result


def test_group_paint_stamps_copies_and_leaves_originals_untouched():
    c0, c1 = _chain(0), _chain(1)
    result = {"chains": [c0, c1]}
    groups = {"fault_a": {"signature": "sig", "chains": [0], "color": [10, 20, 30]}}
    params = _paint_params(spec_json=encode_groups(groups), signature="sig")

    out = GroupPaint().apply(result, params)

    assert out["chains"][0] is not c0                       # copy, not the original
    assert out["chains"][0]["tags"] == ["group:fault_a"]
    assert out["chains"][0]["tag_origin"] == "user:commit"
    assert out["chains"][0]["group_color"] == [10, 20, 30]
    assert out["chains"][1] is c1                            # untouched member, same identity
    assert "tags" not in c0 and "tags" not in c1              # originals never mutated


def test_group_paint_correct_signature_paints_normally():
    c0 = _chain(0)
    result = {"chains": [c0]}
    groups = {"a": {"signature": "sig-1", "chains": [0], "color": [1, 2, 3]}}
    out = GroupPaint().apply(result, _paint_params(spec_json=encode_groups(groups),
                                                    signature="sig-1"))
    assert out["chains"][0]["tags"] == ["group:a"]
    assert "_stale_groups" not in out


def test_group_paint_out_of_range_indices_flag_group_stale_and_are_not_painted():
    c0 = _chain(0)
    result = {"chains": [c0]}
    groups = {"ghost": {"signature": "old-sig", "chains": [5], "color": [1, 2, 3]}}
    out = GroupPaint().apply(result, _paint_params(spec_json=encode_groups(groups)))
    assert out["_stale_groups"] == ["ghost"]
    assert out["chains"][0] is c0            # not painted
    assert "tags" not in c0


# --- GroupPaint: identity-preserving no-op (the scrub-cache bug) ---
#
# `dynamix.shell.canvas.Canvas`'s geometry cache (the 147x scrub fix) keys its trail-polyline
# memoization on `id(result["chains"])`. `apply()` must not copy that list when nothing was
# actually painted -- a copy breaks the identity on every redraw of a committed layer
# regardless of whether anything about the chains had changed. These pin the two honest cases:
# nothing to report at all (full `result` identity, same as the empty-`spec_json` passthrough
# above) vs. something to report but nothing to paint (`_stale_groups` must land on a NEW dict --
# mutating the input in place would break immutability -- but `chains` itself stays identical).


def test_group_paint_empty_groups_object_is_a_full_identity_noop():
    result = {"chains": [_chain(0)]}
    out = GroupPaint().apply(result, _paint_params(spec_json="{}"))
    assert out is result


def test_group_paint_all_stale_preserves_chains_list_identity_not_result_identity():
    c0 = _chain(0)
    result = {"chains": [c0]}
    groups = {"ghost": {"signature": "old-sig", "chains": [5], "color": [1, 2, 3]}}
    out = GroupPaint().apply(result, _paint_params(spec_json=encode_groups(groups)))
    assert out is not result                 # _stale_groups had to land somewhere new
    assert out["chains"] is result["chains"]  # but the LIST itself was never copied
    assert out["_stale_groups"] == ["ghost"]


def test_group_paint_painting_path_still_copies_the_chains_list():
    c0 = _chain(0)
    result = {"chains": [c0]}
    groups = {"a": {"signature": "s", "chains": [0], "color": [1, 2, 3]}}
    out = GroupPaint().apply(result, _paint_params(spec_json=encode_groups(groups)))
    assert out["chains"] is not result["chains"]     # immutability: painting always copies


def test_group_paint_negative_index_is_treated_as_stale():
    c0 = _chain(0)
    result = {"chains": [c0]}
    groups = {"bad": {"signature": "s", "chains": [-1], "color": [1, 1, 1]}}
    out = GroupPaint().apply(result, _paint_params(spec_json=encode_groups(groups)))
    assert out["_stale_groups"] == ["bad"]
    assert "tags" not in c0


def test_group_paint_paints_fresh_group_and_flags_stale_group_independently():
    c0, c1 = _chain(0), _chain(1)
    result = {"chains": [c0, c1]}
    groups = {
        "fresh": {"signature": "s", "chains": [0], "color": [1, 1, 1]},
        "stale": {"signature": "s", "chains": [9], "color": [2, 2, 2]},
    }
    out = GroupPaint().apply(result, _paint_params(spec_json=encode_groups(groups)))
    assert out["chains"][0]["tags"] == ["group:fresh"]
    assert out["_stale_groups"] == ["stale"]


def test_group_paint_chain_can_belong_to_two_groups():
    c0 = _chain(0)
    result = {"chains": [c0]}
    groups = {
        "a": {"signature": "s", "chains": [0], "color": [1, 1, 1]},
        "b": {"signature": "s", "chains": [0], "color": [2, 2, 2]},
    }
    out = GroupPaint().apply(result, _paint_params(spec_json=encode_groups(groups)))
    assert sorted(out["chains"][0]["tags"]) == ["group:a", "group:b"]


def test_group_paint_empty_group_membership_is_not_stale():
    result = {"chains": [_chain(0)]}
    groups = {"empty": {"signature": "s", "chains": [], "color": [1, 1, 1]}}
    out = GroupPaint().apply(result, _paint_params(spec_json=encode_groups(groups)))
    assert "_stale_groups" not in out
    assert "tags" not in out["chains"][0]
    assert out is result           # nothing to paint, nothing to report: full identity


# --- GroupPaint: malformed spec_json must no-op, never raise -------------

def test_group_paint_malformed_json_no_raise_and_flags_spec_error():
    """Mode (a): text that isn't valid JSON at all -- json.loads itself raises. Must not escape
    apply(): a resolve running this filter every redraw would land the layer in a persistent
    error strip instead of a readable no-op."""
    c0 = _chain(0)
    result = {"chains": [c0]}
    out = GroupPaint().apply(result, _paint_params(spec_json="{not valid json"))
    assert "_spec_error" in out
    assert out["chains"] == [c0]
    assert out["chains"][0] is c0
    assert "tags" not in c0


def test_group_paint_non_object_spec_json_no_raise_and_flags_spec_error():
    """Mode (b): valid JSON, wrong shape -- an array where the schema promises {name: {...}}.
    decode_groups itself does not raise (json.loads succeeds); apply() must still catch the
    shape mismatch before it reaches .items()."""
    c0 = _chain(0)
    result = {"chains": [c0]}
    out = GroupPaint().apply(result, _paint_params(spec_json=json.dumps([1, 2, 3])))
    assert "_spec_error" in out
    assert out["chains"][0] is c0
    assert "tags" not in c0


def test_group_paint_non_int_chain_indices_flag_that_group_stale_not_fatal():
    """Mode (c): valid dict, but one group's "chains" entries aren't coercible to int. Must be a
    per-group stale flag, not a fatal raise, and not a whole-spec _spec_error either."""
    c0 = _chain(0)
    result = {"chains": [c0]}
    spec_json = json.dumps({"bad": {"signature": "s", "chains": ["a"], "color": [1, 1, 1]}})
    out = GroupPaint().apply(result, _paint_params(spec_json=spec_json))
    assert out["_stale_groups"] == ["bad"]
    assert "_spec_error" not in out
    assert out["chains"][0] is c0
    assert "tags" not in c0


def test_group_paint_mixed_wellformed_and_garbage_group_paints_fresh_flags_garbage():
    """One well-formed group alongside one structurally-garbage group (not even a mapping): the
    good group must still paint, the garbage one must be flagged stale, and nothing may raise."""
    c0, c1 = _chain(0), _chain(1)
    result = {"chains": [c0, c1]}
    spec_json = json.dumps({
        "fresh": {"signature": "s", "chains": [0], "color": [1, 2, 3]},
        "garbage": "not-a-mapping",
    })
    out = GroupPaint().apply(result, _paint_params(spec_json=spec_json))
    assert out["chains"][0]["tags"] == ["group:fresh"]
    assert out["chains"][0]["group_color"] == [1, 2, 3]
    assert out["_stale_groups"] == ["garbage"]
    assert out["chains"][1] is c1


def test_group_paint_malformed_color_defaults_to_black_without_raising():
    """Minor: a color that isn't exactly 3 numeric entries defaults rather than raising or
    staling the group -- color is display-only, never correctness-bearing."""
    c0 = _chain(0)
    result = {"chains": [c0]}
    spec_json = json.dumps({"a": {"signature": "s", "chains": [0], "color": [1, 2]}})
    out = GroupPaint().apply(result, _paint_params(spec_json=spec_json))
    assert "_stale_groups" not in out
    assert out["chains"][0]["group_color"] == [0, 0, 0]


def test_group_paint_non_numeric_color_defaults_to_black_without_raising():
    c0 = _chain(0)
    result = {"chains": [c0]}
    spec_json = json.dumps({"a": {"signature": "s", "chains": [0], "color": ["r", "g", "b"]}})
    out = GroupPaint().apply(result, _paint_params(spec_json=spec_json))
    assert out["chains"][0]["group_color"] == [0, 0, 0]


# --- GroupFilter ---------------------------------------------------------

def test_group_filter_empty_group_is_passthrough_identity():
    result = {"chains": [_tagged_chain(["group:a"])]}
    out = GroupFilter().apply(result, _filter_params())
    assert out is result


def test_group_filter_keep_mode_keeps_only_matching():
    a, b = _tagged_chain(["group:a"]), _tagged_chain(["group:b"])
    result = {"chains": [a, b]}
    out = GroupFilter().apply(result, _filter_params(group="a", mode="keep"))
    assert out["chains"] == [a]
    assert out["_chains_dropped"] == 1


def test_group_filter_drop_mode_drops_matching():
    a, b = _tagged_chain(["group:a"]), _tagged_chain(["group:b"])
    result = {"chains": [a, b]}
    out = GroupFilter().apply(result, _filter_params(group="a", mode="drop"))
    assert out["chains"] == [b]
    assert out["_chains_dropped"] == 1


def test_group_filter_untagged_chain_never_matches():
    untagged = {"id": 0}
    result = {"chains": [untagged]}
    out = GroupFilter().apply(result, _filter_params(group="a", mode="keep"))
    assert out["chains"] == []


def test_group_filter_group_param_is_editable_group_paint_spec_json_is_not():
    """``group_filter.group`` is the one TEXT param a user must set by
    hand (unlike group_paint's own spec_json/signature, which the commit transaction writes) --
    it must declare ``editable=True`` and earn a real control, not the default read-only label."""
    group_param = next(p for p in GroupFilter.params if p.name == "group")
    assert group_param.editable is True
    assert control_spec(group_param).widget == "text_edit"

    spec_json_param = next(p for p in GroupPaint.params if p.name == "spec_json")
    assert spec_json_param.editable is False
    assert control_spec(spec_json_param).widget == "label"


def test_group_filter_editable_control_propagates_the_typed_group_name(qtbot):
    from dynamix.shell.knob_widgets import make_control

    group_param = next(p for p in GroupFilter.params if p.name == "group")
    w = make_control(control_spec(group_param), "")
    qtbot.addWidget(w)

    w.setText("fault_b")
    with qtbot.waitSignal(w.valueChanged, timeout=1000) as sig:
        w.editingFinished.emit()

    assert sig.args[0] == "fault_b"


# --- registration ---------------------------------------------------------

def test_group_devices_are_registered(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES

    register_builtin_devices()
    assert "group_paint" in DEVICES
    assert "group_filter" in DEVICES
