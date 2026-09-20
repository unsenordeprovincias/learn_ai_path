from dataclasses import FrozenInstanceError, fields

import pytest

from mnist.domain.models import Vector
from mnist.domain.samples import NetSample, RawSample


def test_raw_sample_tiene_los_campos_x_e_y_true_en_ese_orden():
    assert [f.name for f in fields(RawSample)] == ["x", "y_true"]


def test_raw_sample_es_inmutable():
    sample = RawSample(x=b"\x01", y_true=b"\x02")

    with pytest.raises(FrozenInstanceError):
        sample.x = b"\x03"  # type: ignore[misc]


def test_net_sample_es_inmutable():
    sample = NetSample(x=Vector[0, 1], y_true=Vector[1])  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        sample.y_true = Vector[0]  # type: ignore[misc]
