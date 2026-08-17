
from .bten import initialize_op_registry
from .bten.op import make_op as make_op
from .bten.tensor import make_tensor as make_tensor

initialize_op_registry()
