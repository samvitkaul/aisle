
from enum import Enum, auto


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
