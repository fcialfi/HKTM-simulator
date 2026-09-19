"""Named scenario presets: common combinations of *encoding* parameters
(modulation, RS/convolutional FEC, pulse shaping), loaded from presets.json
(next to this file) so adding or tweaking a scenario is a data edit, not a
code change -- and shared between the CLI (`generate_signal.py --preset`)
and the GUI (sidebar selector) so they can't drift apart, same reasoning as
`pipeline.py` being shared by both.

Presets deliberately don't cover payload/duration (n_cadu, seed, payload
source) or output format (dtype, peak, resampling): those describe what to
test with and how to package the result, not how the signal is coded, and
stay independent of which preset is picked.
"""

import json
from pathlib import Path
from typing import TypedDict


class Preset(TypedDict):
    description: str
    modulation: str
    symbol_rate: int
    sps: int
    rrc_alpha: float
    rrc_span: int
    rs_e: int
    interleave_depth: int
    conv_rate: str
    randomizer: str


_REQUIRED_KEYS = set(Preset.__annotations__)
_PRESETS_PATH = Path(__file__).with_name("presets.json")


def _load_presets(path: Path) -> dict[str, Preset]:
    """Load and validate presets.json. Raises early and clearly (at import
    time, so a broken file fails on startup rather than the first time a
    particular preset happens to be picked) rather than deferring to
    whatever cryptic KeyError a missing field would cause deep inside
    ChainParams construction later."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"presets file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object of {{name: preset}}, got {type(data).__name__}")

    for name, cfg in data.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"preset {name!r} in {path} must be a JSON object, got {type(cfg).__name__}")
        missing = _REQUIRED_KEYS - cfg.keys()
        if missing:
            raise ValueError(f"preset {name!r} in {path} is missing required field(s): {sorted(missing)}")
        extra = cfg.keys() - _REQUIRED_KEYS
        if extra:
            raise ValueError(f"preset {name!r} in {path} has unknown field(s): {sorted(extra)}")

    return data


PRESETS: dict[str, Preset] = _load_presets(_PRESETS_PATH)
