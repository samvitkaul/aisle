
from typing import TypeVar, TYPE_CHECKING, Any
from enum import Enum, auto
from functools import lru_cache
from loguru import logger
from copy import deepcopy
import numpy as np
import yaml
import json
import pickle

from pydantic import BaseModel

BaseModel_SubType = TypeVar('BaseModel_SubType', bound=BaseModel)
NUMPY_ARRAY_SIZE_THRESHOLD = 100

class OutFormat(Enum):
    FMT_NONE = auto()
    FMT_YAML = auto()
    FMT_JSON = auto()
    FMT_PICKLE = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return OutFormat['FMT_' + s.upper()]

    @property
    @lru_cache(4)
    def cname(self)->str:
        return self.name.replace('FMT_', '').lower()

def _process_np_attr(v: np.ndarray, op_index: int, opstats: Any, k: str) -> None:
    """Process a numpy array attribute for JSON serialization.

    Converts numpy arrays to descriptive strings containing shape, dtype, and value
    information (truncated for large arrays as needed)

    Args:
        v: The numpy array to process.
        op_index: The index of the operator in the model.
        opstats: The operator statistics object containing the attribute.
        k: The key of the attribute in opstats.attrs.
    """
    # Truncate large arrays for logging
    if v.size > NUMPY_ARRAY_SIZE_THRESHOLD:
        truncated = np.array2string(v, threshold=10, edgeitems=3)
        value_for_output = f"shape={v.shape}, dtype={v.dtype}, truncated: {truncated}"
    else:
        value_for_output = f"shape={v.shape}, dtype={v.dtype}, value={v.tolist()}"
    if opstats.optype in ['Constant', 'ConstantOfShape']:
        logger.warning(
            f"Unexpected numpy.ndarray value for operator op#{op_index} "
            f"(opname {opstats.opname}, optype {opstats.optype}): {k} = {value_for_output}"
        )
    opstats.attrs[k] = value_for_output

def _prepare_model_for_json(model: BaseModel_SubType) -> BaseModel_SubType:
    """Prepare a Pydantic model for JSON serialization by handling numpy arrays.

    Checks for numpy arrays in opstats attributes and creates a deep copy
    if any are found to avoid mutating the original model. Then processes each
    numpy array to make it JSON-serializable.

    Args:
        model: The Pydantic BaseModel to prepare.

    Returns:
        The prepared model (original or deep copy) with numpy arrays converted.
    """

    if not hasattr(model, 'opstats'):
        return model

    has_numpy_arrays = any(
        isinstance(v, np.ndarray)
        for opstats in model.opstats
        if hasattr(opstats, 'attrs')
        for v in opstats.attrs.values()
    )
    if not has_numpy_arrays:
        return model

    model_to_dump = deepcopy(model)
    if TYPE_CHECKING:
        assert hasattr(model_to_dump, 'opstats')

    for op_index, opstats in enumerate(model_to_dump.opstats):
        if hasattr(opstats, 'attrs'):
            for k, v in opstats.attrs.items():
                if isinstance(v, np.ndarray):
                    _process_np_attr(v, op_index, opstats, k)
    return model_to_dump

def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays in nested structures
    into plain Python types so YAML/JSON (and Pydantic) can serialize them."""

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, dict):
        return {_to_builtin(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_builtin(v) for v in obj)
    if isinstance(obj, set):
        return [_to_builtin(v) for v in obj]
    return obj

def dump_model(model: BaseModel, filename, outputfmt: OutFormat)->None:
    if outputfmt == OutFormat.FMT_NONE:
        return
    elif outputfmt == OutFormat.FMT_YAML:
        with open(filename, 'w') as fout:
            data = model.model_dump()
            data = _to_builtin(data)
            yaml.dump(data, fout, indent=4, Dumper=yaml.CDumper)
    elif outputfmt == OutFormat.FMT_JSON:
        model_to_dump = _prepare_model_for_json(model)
        data = model_to_dump.model_dump()
        data = _to_builtin(data)
        with open(filename, 'w') as fout:
            json.dump(data, fout, indent=4, default=str)
    elif outputfmt == OutFormat.FMT_PICKLE:
        with open(filename, 'wb') as foutbin:
            pickle.dump(model, foutbin)
    else:
        raise ValueError(f'Unknown OutFormat: {outputfmt}!!')

