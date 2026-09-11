"""Unit tests for the additive ``value_info`` change to :func:`graph2onnx`.

pins the TENSOR-SHAPE-COMPLETE invariant on the
exporter side: every non-IO tensor in ``G._tensors`` must appear as a
``ValueInfoProto`` in ``graph.value_info`` so the loader can rehydrate
shapes without calling shape inference.
"""
import os

import onnx
import pytest

from src.graph import graph2onnx
from src.graph.graph2onnx import _dim_to_onnx

from ._fixtures import ALL_GRAPH_FIXTURES


def _io_tensor_names(G):
    return set(G.get_input_tensors()) | set(G.get_output_tensors())


def _non_io_activation_tensors(G):
    io = _io_tensor_names(G)
    out = []
    for tname, tval in G._tensors.items():
        if tval.is_const or tval.is_param:
            continue
        if tname in io:
            continue
        out.append(tname)
    return out


@pytest.mark.parametrize(
    'fixture',
    ALL_GRAPH_FIXTURES,
    ids=[f.__name__ for f in ALL_GRAPH_FIXTURES],
)
def test_export_emits_value_info_for_all_intermediates(fixture, tmp_path):
    G = fixture()
    fpath = str(tmp_path / 'g.onnx')
    graph2onnx(G, fpath, do_model_check=False)
    assert os.path.exists(fpath)
    model = onnx.load(fpath, load_external_data=False)

    vi_names = {vi.name for vi in model.graph.value_info}
    expected = set(_non_io_activation_tensors(G))
    missing = expected - vi_names
    assert not missing, (
        f"{fixture.__name__}: intermediate tensors missing from "
        f"graph.value_info: {sorted(missing)}"
    )

    # Inputs / outputs / initializers must NOT also be in value_info.
    io_or_init = _io_tensor_names(G) | {
        n for n, t in G._tensors.items() if t.is_const
    }
    overlap = vi_names & io_or_init
    assert not overlap, (
        f"{fixture.__name__}: value_info overlaps with IO/initializer: "
        f"{sorted(overlap)}"
    )


@pytest.mark.parametrize(
    'fixture',
    ALL_GRAPH_FIXTURES,
    ids=[f.__name__ for f in ALL_GRAPH_FIXTURES],
)
def test_export_value_info_shapes_match_graph(fixture, tmp_path):
    G = fixture()
    fpath = str(tmp_path / 'g.onnx')
    graph2onnx(G, fpath, do_model_check=False)
    model = onnx.load(fpath, load_external_data=False)

    vi_by_name = {vi.name: vi for vi in model.graph.value_info}
    for tname in _non_io_activation_tensors(G):
        vi = vi_by_name[tname]
        expected = [_dim_to_onnx(d) for d in G._tensors[tname].shape]
        actual = []
        for dim in vi.type.tensor_type.shape.dim:
            if dim.HasField('dim_param'):
                actual.append(dim.dim_param)
            else:
                actual.append(int(dim.dim_value))
        assert actual == expected, (
            f"{fixture.__name__}: value_info[{tname!r}] shape {actual} "
            f"does not match graph shape {expected}"
        )
