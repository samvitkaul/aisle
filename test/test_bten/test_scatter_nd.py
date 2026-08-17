
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor
from src.utils.data_types import DataType


def _run_scatternd(
        name,
        data_shape,
        indices_shape,
        updates_shape,
        data_dtype="float32",
        indices_dtype="int64",
        updates_dtype=None,
        ):

    updates_dtype = updates_dtype or data_dtype
    i_tensors = [
            make_tensor(name='D', shape=data_shape,    dtype=data_dtype   ),
            make_tensor(name='I', shape=indices_shape, dtype=indices_dtype),
            make_tensor(name='U', shape=updates_shape, dtype=updates_dtype),
            ]
    o_tensors = [make_tensor(name="Y")]
    op_info = {
        "name": name,
        "optype": "ScatterND",
        "inList": [x.name for x in i_tensors],
        "outList": [x.name for x in o_tensors],
    }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [name]
    for x in o_tensors: x.op_out = [name]
    op_obj(i_tensors, o_tensors)
    return o_tensors[0]

class TestScatterNDShapeInf:

    @pytest.mark.unit
    def test_rank2_k1(self):
        out = _run_scatternd("sc_t1", data_shape=[10, 8], indices_shape=[4, 1], updates_shape=[4, 8])
        assert out.shape == [10, 8]
        assert out.dtype == DataType.FLOAT32

    @pytest.mark.unit
    def test_rank2_k2(self):
        out = _run_scatternd("sc_t1b", data_shape=[10, 8], indices_shape=[3, 2], updates_shape=[3])
        assert out.shape == [10, 8]
        assert out.dtype == DataType.FLOAT32

    @pytest.mark.unit
    def test_rank3_k1(self):
        out = _run_scatternd("sc_t2", data_shape=[6, 4, 8], indices_shape=[3, 2, 1], updates_shape=[3, 2, 4, 8])
        assert out.shape == [6, 4, 8]
        assert out.dtype == DataType.FLOAT32

    @pytest.mark.unit
    def test_dtype_preserved_bf16(self):
        out = _run_scatternd("sc_dt", data_shape=[10, 8], indices_shape=[4, 1], updates_shape=[4, 8],
                             data_dtype='bfloat16', updates_dtype='bfloat16')
        assert out.shape == [10, 8]
        assert out.dtype == DataType.BFLOAT16

    @pytest.mark.unit
    def test_indices_dtype_int64(self):
        with pytest.raises((TypeError, AssertionError)):
            out = _run_scatternd("sc_t3", data_shape=[10, 8], indices_shape=[4, 1], updates_shape=[4, 8],
                             indices_dtype='int32')

    @pytest.mark.unit
    def test_updates_shape_mismatch(self):
        with pytest.raises((ValueError, AssertionError)):
            out = _run_scatternd("sc_t4", data_shape=[10, 8], indices_shape=[4, 1], updates_shape=[4, 7])

    @pytest.mark.unit
    def test_updates_rank_mismatch(self):
        with pytest.raises((ValueError, AssertionError)):
            out = _run_scatternd("sc_t4b", data_shape=[10, 8], indices_shape=[4, 1], updates_shape=[4, 8, 3])

    @pytest.mark.unit
    def test_updates_dtype_mismatch(self):
        with pytest.raises((ValueError, AssertionError)):
            out = _run_scatternd("sc_dt_mis", data_shape=[10, 8], indices_shape=[4, 1], updates_shape=[4, 8],
                                 data_dtype='float32', updates_dtype='bfloat16')

    @pytest.mark.unit
    def test_k_out_of_range(self):
        with pytest.raises((ValueError, AssertionError)):
            out = _run_scatternd("sc_kbig", data_shape=[10, 8], indices_shape=[4, 3], updates_shape=[4])

