
from typing import ClassVar

from loguru import logger
from pydantic import BaseModel, Field, PositiveInt

from ..interconnect import _RackFabricUnion
from . import CompositeSystemMixin
from .blade import Blade
from .nic import NIC

INFO    = logger.info
DEBUG   = logger.debug
ERROR   = logger.error
WARNING = logger.warning

class Rack(CompositeSystemMixin, BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    """
    Specification of a Rack composing multiple Blades.

    A rack is a physical form factor that contains multiple blade servers
    connected via an intra-rack fabric (switch). It models the composition
    of blades and the fabric properties between them.

    Attributes:
        name: Unique identifier for this rack specification
        blade: Reference to the Blade specification used in this rack
        num_blades: Number of blades in the rack
        blade_interconnect: Fabric specification for blade-to-blade connectivity
        cooling_capacity_kW: Total thermal dissipation capacity (optional)
        power_capacity_kW: Total electrical power capacity (optional)
        management_network: Optional management network (OOB) specification
    """
    _child_attr: ClassVar[str] = 'blade'
    _count_attr: ClassVar[str] = 'num_blades'

    name                : str
    blade               : Blade
    num_blades          : PositiveInt
    blade_interconnect  : _RackFabricUnion
    cooling_capacity_kW : float | None = Field(default=None, gt=0.0)
    power_capacity_kW   : float | None = Field(default=None, gt=0.0)
    management_network  : NIC | None = None

    # Task 039 N3: discriminated-union routes raw dicts via ``fabric_kind``.

