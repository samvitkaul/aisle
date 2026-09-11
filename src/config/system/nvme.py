
from loguru import logger
from pydantic import BaseModel

from ..interconnect import PCIe
from ..knob import KnobVal

INFO    = logger.info

class NVME(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    name  : str
    size  : KnobVal
    pcie  : PCIe
