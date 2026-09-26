# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import json

from dynamix.shell.settings import (Settings, load_settings, save_settings, settings_path,
                                    update_settings)


def test_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    s = load_settings()
    assert s.auto_run_wtmm is False


def test_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    save_settings(Settings(auto_run_wtmm=True))
    assert load_settings().auto_run_wtmm is True
    assert json.loads((tmp_path / "s.json").read_text())["auto_run_wtmm"] is True


def test_corrupt_file_yields_defaults(tmp_path, monkeypatch):
    p = tmp_path / "s.json"; p.write_text("{not json")
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(p))
    assert load_settings().auto_run_wtmm is False


def test_new_fields_default_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    s = load_settings()
    assert s.splitter_sizes == {} and s.view_options == {}
    save_settings(Settings(auto_run_wtmm=True,
                           splitter_sizes={"work": [200, 900, 260]},
                           view_options={"graticule": True}))
    s2 = load_settings()
    assert s2.splitter_sizes == {"work": [200, 900, 260]}
    assert s2.view_options == {"graticule": True}
    assert s2.auto_run_wtmm is True


def test_update_settings_preserves_other_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    save_settings(Settings(splitter_sizes={"work": [1, 2, 3]}))
    from dynamix.shell.settings import update_settings
    out = update_settings(auto_run_wtmm=True)
    assert out.auto_run_wtmm is True
    assert load_settings().splitter_sizes == {"work": [1, 2, 3]}   # not clobbered


def test_load_tolerates_garbage_new_fields(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(p))
    p.write_text('{"splitter_sizes": "nope", "view_options": 7}')
    s = load_settings()                       # never raises (the :23-24 contract)
    assert s.splitter_sizes == {} and s.view_options == {}


def test_open_on_launch_defaults_to_none_and_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    assert load_settings().open_on_launch is None
    update_settings(open_on_launch="/tmp/some/raster.tif")
    assert load_settings().open_on_launch == "/tmp/some/raster.tif"


def test_open_window_px_defaults_to_4096_and_rejects_garbage(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    assert load_settings().open_window_px == 4096
    update_settings(open_window_px=1024)
    assert load_settings().open_window_px == 1024
    (tmp_path / "s.json").write_text(json.dumps({"open_window_px": "big"}))
    assert load_settings().open_window_px == 4096
    (tmp_path / "s.json").write_text(json.dumps({"open_window_px": 0}))
    assert load_settings().open_window_px == 4096


def test_footprint_folders_default_empty_and_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    assert load_settings().footprint_folders == []
    update_settings(footprint_folders=["/Volumes/X/DynamiX_Ext", "/data/other"])
    assert load_settings().footprint_folders == ["/Volumes/X/DynamiX_Ext", "/data/other"]
    (tmp_path / "s.json").write_text(json.dumps({"footprint_folders": "not-a-list"}))
    assert load_settings().footprint_folders == []


def test_compute_engine_defaults_validates_and_roundtrips(tmp_path, monkeypatch):
    """Master engine setting: default 'auto'; garbage falls back to 'auto';
    legal values round-trip through update_settings without touching other fields."""
    import json

    from dynamix.shell.settings import load_settings, settings_path, update_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    assert load_settings().compute_engine == "auto"
    # The CPU engine is FFTW3;
    # a settings file saved with the old "numpy" choice loads as that CPU engine, "fftw".
    update_settings(compute_engine="numpy")
    assert load_settings().compute_engine == "fftw"
    update_settings(compute_engine="fftw")
    assert load_settings().compute_engine == "fftw"
    d = json.loads(settings_path().read_text())
    d["compute_engine"] = "cuda-someday"
    settings_path().write_text(json.dumps(d))
    assert load_settings().compute_engine == "auto"


def test_open_max_pixels_defaults_to_64m_and_rejects_garbage(tmp_path, monkeypatch):
    """The ONE threshold above which any raster opens as the picture."""
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    assert load_settings().open_max_pixels == 64_000_000
    update_settings(open_max_pixels=1_000_000)
    assert load_settings().open_max_pixels == 1_000_000
    (tmp_path / "s.json").write_text(json.dumps({"open_max_pixels": "big"}))
    assert load_settings().open_max_pixels == 64_000_000


def test_compute_precision_defaults_to_32_and_round_trips_64(tmp_path, monkeypatch):
    """32 or 64-bit FFTs, 32 by default (xsmurf's own single precision); 64-bit
    runs on FFTW3. Restart to apply."""
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    assert load_settings().compute_precision == 32
    update_settings(compute_precision=64)
    assert load_settings().compute_precision == 64
    (tmp_path / "s.json").write_text(json.dumps({"compute_precision": 16}))
    assert load_settings().compute_precision == 32


def test_beta_keeps_its_own_settings_file_apart_from_dynamix(monkeypatch):
    """GeoDynamix_Beta and DynamiX can live on one Mac; they must never share a settings file."""
    monkeypatch.delenv("DYNAMIX_SETTINGS_PATH", raising=False)
    monkeypatch.setattr("sys.platform", "darwin")
    p = settings_path()
    assert p.parent.name == "GeoDynamix" and p.name == "settings.json"
    assert p.parent.parent.name == "Application Support"


def test_windows_settings_live_under_appdata(monkeypatch, tmp_path):
    monkeypatch.delenv("DYNAMIX_SETTINGS_PATH", raising=False)
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert settings_path() == tmp_path / "Roaming" / "GeoDynamix" / "settings.json"
