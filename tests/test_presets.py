"""Regression tests for named scenario presets: guards against a preset
silently rotting into an invalid or inconsistent configuration as the
chain evolves (e.g. a future change to valid RS-E/interleave-depth
combinations), since generate_signal.py and app.py otherwise only exercise
whichever preset a human happens to click/pass."""

import pytest

from ccsds_chain.mapping import BITS_PER_SYMBOL
from ccsds_chain.pipeline import ChainParams, run_chain
from ccsds_chain.presets import PRESETS

_REQUIRED_KEYS = {
    "description", "modulation", "symbol_rate", "sps", "rrc_alpha",
    "rrc_span", "rs_e", "interleave_depth", "conv_rate", "randomizer",
}


def test_at_least_one_preset_defined():
    assert len(PRESETS) >= 1


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_has_all_required_fields(name):
    assert _REQUIRED_KEYS.issubset(PRESETS[name].keys())
    assert PRESETS[name]["description"]  # non-empty


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_runs_through_the_chain(name):
    cfg = PRESETS[name]
    params = ChainParams(
        modulation=cfg["modulation"],
        bit_rate=cfg["symbol_rate"] * BITS_PER_SYMBOL[cfg["modulation"]],
        symbol_rate=cfg["symbol_rate"],
        sps=cfg["sps"],
        rrc_alpha=cfg["rrc_alpha"],
        rrc_span=cfg["rrc_span"],
        rs_e=cfg["rs_e"],
        interleave_depth=cfg["interleave_depth"],
        conv_rate=cfg["conv_rate"],
        randomizer=cfg["randomizer"],
        n_cadu=3,
        seed=1,
    )
    result = run_chain(params)
    assert len(result.iq) > 0
    assert len(result.symbols) > 0
    assert result.meta["randomizer"] == cfg["randomizer"]
    assert result.meta["conv_rate"] == cfg["conv_rate"]
    assert result.meta["modulation"] == cfg["modulation"]
