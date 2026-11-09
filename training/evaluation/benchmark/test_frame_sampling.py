"""Regression tests for benchmark frame resampling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.benchmark.common import (  # noqa: E402
    _build_deterministic_frame_indices,
    _build_uniform_frame_indices,
)


def test_fps_upsample_repeats_nearest_source_frame():
    """A 20 fps clip resampled to 24 fps can need more frames than the source has.

    14 source frames last 0.65 s. Seventeen 24 fps frames last 0.667 s, which is
    the 8n+1 count used for that duration. The sampler must repeat nearest
    source frames instead of asking the uniform picker for frames that do not exist.
    """
    indices = _build_deterministic_frame_indices(14, 17, 20.0, 24.0)

    assert indices == [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 12, 13]
    assert len(indices) == 17
    assert min(indices) >= 0 and max(indices) < 14


def test_short_source_with_enough_frames_stays_uniform():
    """Keep the uniform spread when the clip is short in time but not in frames."""
    indices = _build_deterministic_frame_indices(30, 25, 60.0, 24.0)

    assert indices == _build_uniform_frame_indices(30, 25)


def test_matching_fps_keeps_the_source_timeline():
    assert _build_deterministic_frame_indices(17, 17, 24.0, 24.0) == list(range(17))
