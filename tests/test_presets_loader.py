"""Regression tests for presets.py's JSON loader itself (separate from
test_presets.py, which exercises the *content* of the real presets.json
against the chain) -- validation and error reporting for a hand-edited
presets file that's missing a field, has an extra one, or isn't valid JSON
at all, since presets.json is meant to be edited by hand."""

import json

import pytest

from ccsds_chain.presets import _load_presets, _REQUIRED_KEYS


def _write(tmp_path, content):
    path = tmp_path / "presets.json"
    path.write_text(content)
    return path


def _valid_preset(**overrides):
    preset = {
        "description": "test",
        "modulation": "QPSK",
        "symbol_rate": 1_785_000,
        "sps": 4,
        "rrc_alpha": 0.35,
        "rrc_span": 8,
        "rs_e": 16,
        "interleave_depth": 5,
        "conv_rate": "1/2",
        "randomizer": "long",
    }
    preset.update(overrides)
    return preset


class TestLoader:
    def test_loads_a_valid_file(self, tmp_path):
        path = _write(tmp_path, json.dumps({"mine": _valid_preset()}))
        loaded = _load_presets(path)
        assert loaded == {"mine": _valid_preset()}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_presets(tmp_path / "does_not_exist.json")

    def test_invalid_json_raises(self, tmp_path):
        path = _write(tmp_path, "{not valid json")
        with pytest.raises(ValueError, match="not valid JSON"):
            _load_presets(path)

    def test_top_level_must_be_an_object(self, tmp_path):
        path = _write(tmp_path, json.dumps(["not", "an", "object"]))
        with pytest.raises(ValueError, match="JSON object"):
            _load_presets(path)

    def test_preset_must_be_an_object(self, tmp_path):
        path = _write(tmp_path, json.dumps({"mine": "not an object"}))
        with pytest.raises(ValueError, match="must be a JSON object"):
            _load_presets(path)

    def test_missing_field_is_reported_by_name(self, tmp_path):
        preset = _valid_preset()
        del preset["rrc_alpha"]
        path = _write(tmp_path, json.dumps({"mine": preset}))
        with pytest.raises(ValueError, match="rrc_alpha"):
            _load_presets(path)

    def test_unknown_field_is_reported_by_name(self, tmp_path):
        preset = _valid_preset(typo_field=1)
        path = _write(tmp_path, json.dumps({"mine": preset}))
        with pytest.raises(ValueError, match="typo_field"):
            _load_presets(path)

    def test_required_keys_match_every_field_a_valid_preset_has(self):
        assert _REQUIRED_KEYS == set(_valid_preset().keys())
