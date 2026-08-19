
from ..interconnect import PCIe
from ..knob import KnobVal

from pydantic import BaseModel
from loguru   import logger

INFO    = logger.info

class NIC(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    name      : str
    bandwidth : KnobVal
    pcie      : PCIe
