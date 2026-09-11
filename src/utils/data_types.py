
from enum import Enum, auto

import ml_dtypes as mlnp
import numpy as np


class DataType(Enum):
    BOOL                = auto()
    INT2                = auto()
    INT4                = auto()
    INT8                = auto()
    INT16               = auto()
    INT32               = auto()
    INT64               = auto()
    UINT2               = auto()
    UINT4               = auto()
    UINT8               = auto()
    UINT16              = auto()
    UINT32              = auto()
    UINT64              = auto()
    FLOAT8              = auto()
    FLOAT8_e3m4         = auto()
    FLOAT8_e4m3         = auto()
    FLOAT8_e4m3b11fnuz  = auto()
    FLOAT8_e4m3fn       = auto()
    FLOAT8_e4m3fnuz     = auto()
    FLOAT8_e5m2         = auto()
    FLOAT8_e5m2fnuz     = auto()
    FLOAT8_e8m0fnu      = auto()
    FLOAT4_e2m1fn       = auto()
    FLOAT6_e2m3fn       = auto()
    FLOAT6_e3m2fn       = auto()
    BFLOAT16            = auto()
    TENSOR_FLOAT32      = auto()
    FLOAT16             = auto()
    FLOAT32             = auto()
    FLOAT64             = auto()
    UNDEF               = auto()

    def __str__(self):
        return self.name


# ----- str to dt -----------------
_STR2DT_TBL = {
        'BOOL'                : DataType.BOOL,
        'INT2'                : DataType.INT2,
        'INT4'                : DataType.INT4,
        'INT8'                : DataType.INT8,
        'INT16'               : DataType.INT16,
        'INT32'               : DataType.INT32,
        'INT64'               : DataType.INT64,
        'UINT2'               : DataType.UINT2,
        'UINT4'               : DataType.UINT4,
        'UINT8'               : DataType.UINT8,
        'UINT16'              : DataType.UINT16,
        'UINT32'              : DataType.UINT32,
        'UINT64'              : DataType.UINT64,
        'FLOAT8'              : DataType.FLOAT8,
        'FLOAT8_E3M4'         : DataType.FLOAT8_e3m4,
        'FLOAT8_E4M3'         : DataType.FLOAT8_e4m3,
        'FLOAT8_E4M3B11FNUZ'  : DataType.FLOAT8_e4m3b11fnuz,
        'FLOAT8_E4M3FN'       : DataType.FLOAT8_e4m3fn,
        'FLOAT8_E4M3FNUZ'     : DataType.FLOAT8_e4m3fnuz,
        'FLOAT8_E5M2'         : DataType.FLOAT8_e5m2,
        'FLOAT8_E5M2FNUZ'     : DataType.FLOAT8_e5m2fnuz,
        'FLOAT8_E8M0FNU'      : DataType.FLOAT8_e8m0fnu,
        'FLOAT4_E2M1FN'       : DataType.FLOAT4_e2m1fn,
        'FLOAT6_E2M3FN'       : DataType.FLOAT6_e2m3fn,
        'FLOAT6_E3M2FN'       : DataType.FLOAT6_e3m2fn,
        'BFLOAT16'            : DataType.BFLOAT16,
        'TENSOR_FLOAT32'      : DataType.TENSOR_FLOAT32,
        'FLOAT16'             : DataType.FLOAT16,
        'FLOAT32'             : DataType.FLOAT32,
        'FLOAT64'             : DataType.FLOAT64,
        #aliases
        'FP4'                 : DataType.FLOAT4_e2m1fn,
        'FP8'                 : DataType.FLOAT8,
        'FP16'                : DataType.FLOAT16,
        'BF16'                : DataType.BFLOAT16,
        'FP32'                : DataType.FLOAT32,
        'FP64'                : DataType.FLOAT64,
        'TF32'                : DataType.TENSOR_FLOAT32,
        }

def str2dt(dtype: str) -> DataType:
    try:
        dt = _STR2DT_TBL[dtype.upper()]
    except KeyError:
        dt = DataType.UNDEF
    return dt

# ----- dt fallbacks -----------------
def dt_fallbacks(dtype: DataType) -> list[DataType]:
    _tbl = {
            DataType.INT2          : ['INT4', 'INT8', 'INT16', 'INT32', 'INT64'],
            DataType.INT4          : ['INT8', 'INT16', 'INT32', 'INT64'],
            DataType.INT8          : ['INT16', 'INT32', 'INT64'],
            DataType.INT16         : ['INT32', 'INT64'],
            DataType.INT32         : ['INT64'],
            DataType.UINT2         : ['UINT4', 'UINT8', 'UINT16', 'UINT32', 'UINT64'],
            DataType.UINT4         : ['UINT8', 'UINT16', 'UINT32', 'UINT64'],
            DataType.UINT8         : ['UINT16', 'UINT32', 'UINT64'],
            DataType.UINT16        : ['UINT32', 'UINT64'],
            DataType.UINT32        : ['UINT64'],
            DataType.FLOAT8        : ['FLOAT16', 'FLOAT32', 'FLOAT64'],
            DataType.BFLOAT16      : ['FLOAT32', 'FLOAT64'],
            DataType.TENSOR_FLOAT32: ['FLOAT32', 'FLOAT64'],
            DataType.FLOAT16       : ['FLOAT32', 'FLOAT64'],
            DataType.FLOAT32       : ['FLOAT64'],
            DataType.FLOAT4_e2m1fn : ['FLOAT8', 'FLOAT16', 'FLOAT32', 'FLOAT64'],
            }
    try:
        fallback_types = _tbl[dtype]
    except KeyError:
        fallback_types = []
    return [str2dt(x) for x in fallback_types]


# ----- promote types -----------------
_SIGNED_INT_RANK = {
        DataType.INT2  : 0,
        DataType.INT4  : 1,
        DataType.INT8  : 2,
        DataType.INT16 : 3,
        DataType.INT32 : 4,
        DataType.INT64 : 5,
        }

_UNSIGNED_INT_RANK = {
        DataType.UINT2  : 0,
        DataType.UINT4  : 1,
        DataType.UINT8  : 2,
        DataType.UINT16 : 3,
        DataType.UINT32 : 4,
        DataType.UINT64 : 5,
        }

_SUB16_FLOATS = frozenset({
    DataType.FLOAT8,
    DataType.FLOAT8_e3m4,
    DataType.FLOAT8_e4m3,
    DataType.FLOAT8_e4m3b11fnuz,
    DataType.FLOAT8_e4m3fn,
    DataType.FLOAT8_e4m3fnuz,
    DataType.FLOAT8_e5m2,
    DataType.FLOAT8_e5m2fnuz,
    DataType.FLOAT8_e8m0fnu,
    DataType.FLOAT4_e2m1fn,
    DataType.FLOAT6_e2m3fn,
    DataType.FLOAT6_e3m2fn,
    })

_FLOAT_RANK = {
        **{d: 0 for d in _SUB16_FLOATS},
        DataType.FLOAT16        : 1,
        DataType.BFLOAT16       : 1,
        DataType.TENSOR_FLOAT32 : 2,
        DataType.FLOAT32        : 2,
        DataType.FLOAT64        : 3,
        }

_UNSIGNED_TO_MIN_SIGNED = {
        DataType.UINT2  : DataType.INT4,
        DataType.UINT4  : DataType.INT8,
        DataType.UINT8  : DataType.INT16,
        DataType.UINT16 : DataType.INT32,
        DataType.UINT32 : DataType.INT64,
        }

def promote_types(a: DataType, b: DataType) -> DataType:
    if not isinstance(a, DataType) or not isinstance(b, DataType):
        raise TypeError(
                f"promote_types requires DataType inputs, got "
                f"{type(a).__name__}, {type(b).__name__}"
                )

    if a is b:
        return a
    if a is DataType.UNDEF or b is DataType.UNDEF:
        return DataType.UNDEF
    if a is DataType.BOOL:
        return b
    if b is DataType.BOOL:
        return a

    a_sint = a in _SIGNED_INT_RANK
    b_sint = b in _SIGNED_INT_RANK
    a_uint = a in _UNSIGNED_INT_RANK
    b_uint = b in _UNSIGNED_INT_RANK
    a_flt  = a in _FLOAT_RANK
    b_flt  = b in _FLOAT_RANK

    #Int + float -> float, int width discarded;
    # so INT64 + FP16 -> FP16
    if (a_sint or a_uint) and b_flt:
        return b
    if (b_sint or b_uint) and a_flt:
        return a

    if a_sint and b_sint:
        return a if _SIGNED_INT_RANK[a] >= _SIGNED_INT_RANK[b] else b

    if a_uint and b_uint:
        return a if _UNSIGNED_INT_RANK[a] >= _UNSIGNED_INT_RANK[b] else b

    if (a_sint and b_uint) or (b_sint and a_uint):
        signed, unsigned = (a,b) if a_sint else (b,a)
        if unsigned is DataType.UINT64:
            return DataType.FLOAT64
        needed = _UNSIGNED_TO_MIN_SIGNED[unsigned]
        if _SIGNED_INT_RANK[signed] >= _SIGNED_INT_RANK[needed]:
            return signed
        return needed

    if a_flt and b_flt:
        pair = frozenset({a, b})
        if pair == frozenset({DataType.BFLOAT16, DataType.FLOAT16}):
            return DataType.FLOAT32
        if pair == frozenset({DataType.TENSOR_FLOAT32, DataType.FLOAT32}):
            return DataType.FLOAT32
        ra, rb = _FLOAT_RANK[a], _FLOAT_RANK[b]
        if ra != rb:
            return a if ra > rb else b
        if ra == 0:
            return DataType.FLOAT16
        return DataType.FLOAT32

    raise ValueError(f"promote_types: unhandled pair: ({a}, {b})")

# ----- bpe -----------------
_BPE_TBL = {
        DataType.BOOL               : 1,
        DataType.INT8               : 1,
        DataType.INT16              : 2,
        DataType.INT32              : 4,
        DataType.INT64              : 8,
        DataType.UINT8              : 1,
        DataType.UINT16             : 2,
        DataType.UINT32             : 4,
        DataType.UINT64             : 8,
        DataType.BFLOAT16           : 2,
        DataType.TENSOR_FLOAT32     : 4,
        DataType.FLOAT16            : 2,
        DataType.FLOAT32            : 4,
        DataType.FLOAT64            : 32,

        #need a better estimator for these
        DataType.INT2               : 1,
        DataType.INT4               : 1,
        DataType.UINT2              : 1,
        DataType.UINT4              : 1,
        DataType.FLOAT8             : 1,
        DataType.FLOAT8_e3m4        : 1,
        DataType.FLOAT8_e4m3        : 1,
        DataType.FLOAT8_e4m3b11fnuz : 1,
        DataType.FLOAT8_e4m3fn      : 1,
        DataType.FLOAT8_e4m3fnuz    : 1,
        DataType.FLOAT8_e5m2        : 1,
        DataType.FLOAT8_e5m2fnuz    : 1,
        DataType.FLOAT8_e8m0fnu     : 1,
        DataType.FLOAT4_e2m1fn      : 1,
        DataType.FLOAT6_e2m3fn      : 1,
        DataType.FLOAT6_e3m2fn      : 1,

        DataType.UNDEF      : -1,
        }

def get_bpe(dtype: DataType) -> int:
    return _BPE_TBL[dtype]

# ----- dt to np -----------------
_DT2NP_TBL = {
        DataType.BOOL               : np.bool_,
        DataType.INT8               : np.int8,
        DataType.INT16              : np.int16,
        DataType.INT32              : np.int32,
        DataType.INT64              : np.int64,
        DataType.UINT8              : np.uint8,
        DataType.UINT16             : np.uint16,
        DataType.UINT32             : np.uint32,
        DataType.UINT64             : np.uint64,
        DataType.BFLOAT16           : np.float16,
        DataType.TENSOR_FLOAT32     : np.float32,
        DataType.FLOAT16            : np.float16,
        DataType.FLOAT32            : np.float32,
        DataType.FLOAT64            : np.float64,

        #need a better estimator for these
        DataType.INT2               : mlnp.int2,
        DataType.INT4               : mlnp.int4,
        DataType.UINT2              : mlnp.uint2,
        DataType.UINT4              : mlnp.uint4,
        DataType.FLOAT8             : mlnp.float8_e3m4,
        DataType.FLOAT8_e3m4        : mlnp.float8_e3m4,
        DataType.FLOAT8_e4m3        : mlnp.float8_e4m3,
        DataType.FLOAT8_e4m3b11fnuz : mlnp.float8_e4m3b11fnuz,
        DataType.FLOAT8_e4m3fn      : mlnp.float8_e4m3fn,
        DataType.FLOAT8_e4m3fnuz    : mlnp.float8_e4m3fnuz,
        DataType.FLOAT8_e5m2        : mlnp.float8_e5m2,
        DataType.FLOAT8_e5m2fnuz    : mlnp.float8_e5m2fnuz,
        DataType.FLOAT8_e8m0fnu     : mlnp.float8_e8m0fnu,
        DataType.FLOAT4_e2m1fn      : mlnp.float4_e2m1fn,
        DataType.FLOAT6_e2m3fn      : mlnp.float6_e2m3fn,
        DataType.FLOAT6_e3m2fn      : mlnp.float6_e3m2fn,

        }

def dt2np(dtype: DataType) -> np.generic:
    try:
        return _DT2NP_TBL[dtype]
    except KeyError:
        raise KeyError("No numpy type mapping for {dtype}")

