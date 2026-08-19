
from ..knob import KnobVal
from ..interconnect import PCIe

from pydantic import BaseModel
from loguru   import logger

INFO    = logger.info

class NVME(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    name  : str
    size  : KnobVal
    pcie  : PCIe
