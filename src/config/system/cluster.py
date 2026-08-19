
from .rack import Rack
from ..interconnect import _DataCenterFabricUnion
from . import CompositeSystemMixin

from pydantic import BaseModel, PositiveInt
from typing import Any, ClassVar, Dict, Optional
from loguru   import logger

INFO    = logger.info
DEBUG   = logger.debug
ERROR   = logger.error
WARNING = logger.warning

class Cluster(CompositeSystemMixin, BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    """
    Specification of a Cluster composing multiple Racks.

    A cluster is a collection of racks connected via a data center fabric
    (typically a multi-tier switch architecture like Clos or Spine-Leaf).
    It represents a complete deployable unit within a data center.

    Attributes:
        name: Unique identifier for this cluster specification
        rack: Reference to the Rack specification used in this cluster
        num_racks: Number of racks in the cluster
        rack_interconnect: Fabric specification for rack-to-rack connectivity
        orchestration_layer: Optional orchestration system (e.g., "kubernetes", "custom_scheduler")
        region_info: Optional geographic/metadata information about the cluster location
    """
    _child_attr: ClassVar[str] = 'rack'
    _count_attr: ClassVar[str] = 'num_racks'

    name                : str
    rack                : Rack
    num_racks           : PositiveInt
    rack_interconnect   : _DataCenterFabricUnion
    orchestration_layer : Optional[str] = None
    region_info         : Optional[Dict[str, Any]] = None

    # Task 039 N3: discriminated-union routes raw dicts via ``fabric_kind``.

