
import threading

from .data_manip import register_data_manip_ops
from .math import register_math_ops

#from .ccl import register_ccl_ops
from .nn import register_nn_ops
from .reduction import register_reduction_ops

_init_lock = threading.Lock()
_initialized = False

def initialize_op_registry():
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return #type: ignore[unreachable] #double checked locking
        register_math_ops()
        register_nn_ops()
        register_data_manip_ops()
        #register_ccl_ops()
        register_reduction_ops()
        _initialized = True

