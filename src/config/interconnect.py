
from .knob import KnobVal

from pydantic import (
    BaseModel, Discriminator, Field, PositiveInt, TypeAdapter, model_validator,
)
from dataclasses import dataclass
from typing import (
    Annotated, Any, ClassVar, Iterable, Iterator, Literal, Optional,
    Union, get_args, get_origin, get_type_hints,
)

# ---------------------------------------------------------------------------
# NetworkSpec polymorphic base + invariants.
# ---------------------------------------------------------------------------

NetworkTopologyKind = Literal[
    'all_to_all', 'ring', 'mesh', 'hypercube',
    'fat_tree', 'switched', 'point_to_point',
]

def _knob_seconds(k: KnobVal) -> float:
    return k.get('s')

def _knob_Bps(k: KnobVal) -> float:
    try:
        return k.get('B/s')
    except ValueError:
        return k.get('b/s') / 8.0

@dataclass(frozen=True)
class OwnerContext:
    """Lightweight context passed to ``NetworkSpec.member_endpoint_ids``.

    ``owner_path`` is the dotted pydantic-field path to the parent
    container (excluding the spec's own field name). ``owner`` is the
    parent :class:`BaseModel` instance itself (added in N4 so concrete
    subclasses can enumerate sibling endpoints), or ``None`` when the
    spec sits on the system root.
    """
    owner_path: tuple[str, ...]
    owner: Optional[BaseModel] = None


class NetworkSpec(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    """Abstract base for every interconnect that participates in CCL pricing.

    Subclasses implement ``alpha_seconds`` / ``beta_seconds_per_byte`` /
    ``member_endpoint_ids``; ``gamma_seconds_per_byte_peer`` defaults to 0.0
    and ``price_collective`` forwards to the backend's generic ``default_price``
    (wired in Task 039 N1).
    """
    name: str
    role: Literal['compute_to_compute', 'compute_to_memaccel'] = 'compute_to_compute'

    topology_kind: ClassVar[NetworkTopologyKind]

    @model_validator(mode='after')
    def _forbid_abstract_instantiation(self):
        if type(self) is NetworkSpec:
            raise TypeError(
                "NetworkSpec is abstract; instantiate a concrete subclass.")
        return self

    def alpha_seconds(self) -> float:
        raise NotImplementedError

    def beta_seconds_per_byte(self) -> float:
        raise NotImplementedError

    def gamma_seconds_per_byte_peer(self) -> float:
        return 0.0

    def member_endpoint_ids(self, owner_ctx: 'OwnerContext') -> frozenset[str]:
        raise NotImplementedError

    def price_collective(self, ccl_type, msg_bytes, group_size):
        try:
            from src.back.ccl_cost import default_price
        except ImportError as exc:
            raise NotImplementedError(
                "price_collective is wired in N1") from exc
        return default_price(self, ccl_type, msg_bytes, group_size)


class PCIe(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    name     : str
    gen      : str
    link_bw  : KnobVal
    num_links: PositiveInt

    def peak_bandwidth(self, units: str = 'GB/s') -> float:
        """this is unidirectional BW, for bidirectional do a 2x factor"""
        return self.num_links * self.link_bw.get(units)

class NVLink(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    name     : str
    gen      : str
    link_bw  : KnobVal
    num_links: PositiveInt

    def peak_bandwidth(self, units: str = 'GB/s') -> float:
        return self.num_links * self.link_bw.get(units)

# ---------------------------------------------------------------------------
# Task 039 N3: fabric base classes are abstract; concrete subclasses below
# carry a ``fabric_kind`` discriminator literal. Pydantic discriminated
# unions (see system containers + the module-level TypeAdapter helpers) are
# the only sanctioned construction path.
# ---------------------------------------------------------------------------

class RackFabric(NetworkSpec, extra='forbid', populate_by_name=True, frozen=True):
    topology_kind: ClassVar[NetworkTopologyKind] = 'switched'

    switch_model        : str
    bandwidth_per_port  : KnobVal
    latency             : KnobVal
    num_ports           : PositiveInt
    oversubscription    : float = Field(default=1.0, ge=0.1)

    @model_validator(mode='after')
    def _forbid_abstract_rackfabric(self):
        if type(self) is RackFabric:
            raise TypeError(
                "RackFabric is abstract; instantiate a concrete subclass "
                "(NVLinkSwitchRackFabric, PCIeSwitchRackFabric, HybridRackFabric).")
        return self

    def alpha_seconds(self) -> float:
        return _knob_seconds(self.latency)

    def beta_seconds_per_byte(self) -> float:
        bw = _knob_Bps(self.bandwidth_per_port) * self.oversubscription
        if bw <= 0.0:
            raise ValueError(
                f"RackFabric {self.name!r}: non-positive effective bandwidth "
                f"({bw} B/s) from bandwidth_per_port × oversubscription")
        return 1.0 / bw

    def member_endpoint_ids(self, owner_ctx: 'OwnerContext') -> frozenset[str]:
        return frozenset()

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if cls is RackFabric and isinstance(obj, dict) and 'fabric_kind' in obj:
            return _RackFabricAdapter.validate_python(obj)
        return super().model_validate(obj, *args, **kwargs)


class NVLinkSwitchRackFabric(RackFabric, extra='forbid', populate_by_name=True, frozen=True):
    fabric_kind: Literal['nvlink_switch']


class PCIeSwitchRackFabric(RackFabric, extra='forbid', populate_by_name=True, frozen=True):
    fabric_kind: Literal['pcie_switch']


class HybridRackFabric(RackFabric, extra='forbid', populate_by_name=True, frozen=True):
    fabric_kind: Literal['hybrid']


_RackFabricUnion = Annotated[
    Union[NVLinkSwitchRackFabric, PCIeSwitchRackFabric, HybridRackFabric],
    Discriminator('fabric_kind'),
]


class DataCenterFabric(NetworkSpec, extra='forbid', populate_by_name=True, frozen=True):
    topology_kind: ClassVar[NetworkTopologyKind] = 'fat_tree'

    uplink_bandwidth      : KnobVal
    inter_rack_latency    : KnobVal
    link_redundancy       : PositiveInt = Field(default=1, ge=1)
    switch_model          : Optional[str] = None

    @model_validator(mode='after')
    def _forbid_abstract_dcfabric(self):
        if type(self) is DataCenterFabric:
            raise TypeError(
                "DataCenterFabric is abstract; instantiate a concrete subclass "
                "(SpineLeafDataCenterFabric, ClosDataCenterFabric, "
                "FullMeshDataCenterFabric).")
        return self

    def alpha_seconds(self) -> float:
        return _knob_seconds(self.inter_rack_latency)

    def beta_seconds_per_byte(self) -> float:
        bw = _knob_Bps(self.uplink_bandwidth)
        if bw <= 0.0:
            raise ValueError(
                f"DataCenterFabric {self.name!r}: non-positive uplink "
                f"bandwidth ({bw} B/s)")
        return 1.0 / bw

    def member_endpoint_ids(self, owner_ctx: 'OwnerContext') -> frozenset[str]:
        return frozenset()

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if cls is DataCenterFabric and isinstance(obj, dict) and 'fabric_kind' in obj:
            return _DataCenterFabricAdapter.validate_python(obj)
        return super().model_validate(obj, *args, **kwargs)


class SpineLeafDataCenterFabric(DataCenterFabric, extra='forbid', populate_by_name=True, frozen=True):
    fabric_kind: Literal['spine_leaf']


class ClosDataCenterFabric(DataCenterFabric, extra='forbid', populate_by_name=True, frozen=True):
    fabric_kind: Literal['clos']


class FullMeshDataCenterFabric(DataCenterFabric, extra='forbid', populate_by_name=True, frozen=True):
    fabric_kind: Literal['full_mesh']


_DataCenterFabricUnion = Annotated[
    Union[SpineLeafDataCenterFabric, ClosDataCenterFabric, FullMeshDataCenterFabric],
    Discriminator('fabric_kind'),
]


class BladeFabric(NetworkSpec, extra='forbid', populate_by_name=True, frozen=True):
    topology_kind: ClassVar[NetworkTopologyKind] = 'switched'

    bandwidth_per_link: KnobVal
    latency           : KnobVal
    num_links_per_gpu : PositiveInt = Field(default=1, ge=1)
    oversubscription  : float       = Field(default=1.0, ge=0.1)

    @model_validator(mode='after')
    def _forbid_abstract_bladefabric(self):
        if type(self) is BladeFabric:
            raise TypeError(
                "BladeFabric is abstract; instantiate a concrete subclass "
                "(NVSwitchBladeFabric, NVLinkFullMeshBladeFabric, "
                "NVLinkHypercubeBladeFabric, PCIePeerBladeFabric, "
                "HybridBladeFabric).")
        return self

    def alpha_seconds(self) -> float:
        return _knob_seconds(self.latency)

    def beta_seconds_per_byte(self) -> float:
        bw = _knob_Bps(self.bandwidth_per_link) * self.num_links_per_gpu
        if bw <= 0.0:
            raise ValueError(
                f"BladeFabric {self.name!r}: non-positive aggregate bandwidth "
                f"({bw} B/s) from bandwidth_per_link × num_links_per_gpu")
        return 1.0 / bw

    def member_endpoint_ids(self, owner_ctx: 'OwnerContext') -> frozenset[str]:
        return frozenset()

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if cls is BladeFabric and isinstance(obj, dict) and 'fabric_kind' in obj:
            return _BladeFabricAdapter.validate_python(obj)
        return super().model_validate(obj, *args, **kwargs)


class NVSwitchBladeFabric(BladeFabric, extra='forbid', populate_by_name=True, frozen=True):
    """NVSwitch-backed blade fabric.

    Overrides ``price_collective`` so AllReduce is priced as a single
    switch hop (α + β·m) rather than the default switch-ladder formula.
    Other collectives fall through to :func:`default_price`.
    """
    fabric_kind: Literal['nvswitch']
    topology_kind: ClassVar[NetworkTopologyKind] = 'switched'

    def price_collective(self, ccl_type, msg_bytes, group_size):
        from src.back.ccl_cost import str2ccl, CCLType
        try:
            kind = str2ccl(ccl_type) if isinstance(ccl_type, str) else ccl_type
        except Exception:
            return super().price_collective(ccl_type, msg_bytes, group_size)
        if kind is CCLType.AR:
            return self.alpha_seconds() + self.beta_seconds_per_byte() * msg_bytes
        return super().price_collective(ccl_type, msg_bytes, group_size)


class NVLinkFullMeshBladeFabric(BladeFabric, extra='forbid', populate_by_name=True, frozen=True):
    fabric_kind: Literal['nvlink_full_mesh']
    topology_kind: ClassVar[NetworkTopologyKind] = 'all_to_all'


class NVLinkHypercubeBladeFabric(BladeFabric, extra='forbid', populate_by_name=True, frozen=True):
    fabric_kind: Literal['nvlink_hypercube']
    topology_kind: ClassVar[NetworkTopologyKind] = 'hypercube'


class PCIePeerBladeFabric(BladeFabric, extra='forbid', populate_by_name=True, frozen=True):
    fabric_kind: Literal['pcie_peer']
    topology_kind: ClassVar[NetworkTopologyKind] = 'all_to_all'


class HybridBladeFabric(BladeFabric, extra='forbid', populate_by_name=True, frozen=True):
    fabric_kind: Literal['hybrid']
    topology_kind: ClassVar[NetworkTopologyKind] = 'switched'


_BladeFabricUnion = Annotated[
    Union[
        NVSwitchBladeFabric, NVLinkFullMeshBladeFabric,
        NVLinkHypercubeBladeFabric, PCIePeerBladeFabric, HybridBladeFabric,
    ],
    Discriminator('fabric_kind'),
]


# ---------------------------------------------------------------------------
# strict-typing invariant validator.
# ---------------------------------------------------------------------------

def _annotation_classes(ann: Any) -> Iterable[type]:
    """Yield every class mentioned anywhere in a type annotation."""
    if isinstance(ann, type):
        yield ann
        return
    origin = get_origin(ann)
    if origin is not None and isinstance(origin, type):
        yield origin
    for arg in get_args(ann):
        yield from _annotation_classes(arg)


def assert_networkspec_invariants(root: BaseModel) -> None:
    """Walk ``root`` and raise ``ValueError`` on any strict-typing violation.

    See Task 039 §6.1 step 4 and §2.1.9 for the contract. Called from
    ``ExecSystem.__init__``; cheap, runs once per system construction.
    """
    seen: set[int] = set()
    _walk_invariants(root, (), seen)


def _walk_invariants(obj: Any, path: tuple[str, ...], seen: set[int]) -> None:
    if not isinstance(obj, BaseModel):
        return
    if id(obj) in seen:
        return
    seen.add(id(obj))

    cls = type(obj)
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    for fname in cls.__pydantic_fields__:
        ann = hints.get(fname, cls.__pydantic_fields__[fname].annotation)
        value = getattr(obj, fname)
        field_path = '.'.join((*path, fname))
        ann_classes = list(_annotation_classes(ann))
        ann_has_networkspec = any(
            issubclass(c, NetworkSpec) for c in ann_classes
            if isinstance(c, type)
        )
        ann_has_basemodel = any(
            issubclass(c, BaseModel) for c in ann_classes
            if isinstance(c, type)
        )

        if ann_has_networkspec:
            _enforce_networkspec_value(value, field_path)

        _enforce_no_stray_networkspec(value, ann_has_networkspec, field_path)

        if ann_has_basemodel:
            if isinstance(value, BaseModel):
                _walk_invariants(value, (*path, fname), seen)
            elif isinstance(value, (list, tuple)):
                for i, elt in enumerate(value):
                    if isinstance(elt, BaseModel):
                        _walk_invariants(
                            elt, (*path, f"{fname}[{i}]"), seen)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, BaseModel):
                        _walk_invariants(
                            v, (*path, f"{fname}[{k!r}]"), seen)


def _enforce_networkspec_value(value: Any, field_path: str) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for i, elt in enumerate(value):
            if elt is None or isinstance(elt, NetworkSpec):
                continue
            raise ValueError(
                f"field '{field_path}[{i}]' is typed NetworkSpec but holds "
                f"{type(elt).__name__!r}, which is not a NetworkSpec subclass")
        return
    if isinstance(value, NetworkSpec):
        return
    raise ValueError(
        f"field '{field_path}' is typed NetworkSpec but holds "
        f"{type(value).__name__!r}, which is not a NetworkSpec subclass")


def _enforce_no_stray_networkspec(value: Any, ann_has_networkspec: bool,
                                  field_path: str) -> None:
    if ann_has_networkspec:
        return
    if isinstance(value, NetworkSpec):
        raise ValueError(
            f"field '{field_path}' holds a NetworkSpec instance "
            f"({type(value).__name__!r}) but its static annotation does not "
            f"include NetworkSpec; widen the field type to NetworkSpec")
    if isinstance(value, (list, tuple)):
        for i, elt in enumerate(value):
            if isinstance(elt, NetworkSpec):
                raise ValueError(
                    f"field '{field_path}[{i}]' holds a NetworkSpec instance "
                    f"({type(elt).__name__!r}) but its static annotation "
                    f"does not include NetworkSpec; widen the field type")


# ---------------------------------------------------------------------------
# Task 039 N1: walk_networks — generic NetworkSpec discovery walker.
# ---------------------------------------------------------------------------

def walk_networks(
    root: BaseModel,
    *,
    owner_path: tuple[str, ...] = (),
) -> Iterator[tuple[NetworkSpec, 'OwnerContext']]:
    """Yield every :class:`NetworkSpec` reachable from ``root`` with its
    :class:`OwnerContext`.

    Pure recursion over pydantic ``__pydantic_fields__``; no knowledge of
    Blade / Rack / Cluster / Die / vendor. ``isinstance(x, NetworkSpec)``
    is the sole discovery predicate — link primitives (``PCIe``,
    ``NVLink``) are never yielded.

    Each yielded :class:`OwnerContext` carries the parent ``BaseModel``
    instance (N4) so concrete subclasses' ``member_endpoint_ids`` can
    enumerate sibling endpoints. The Nvidia base-class implementations
    return ``frozenset()`` until vendor-specific overrides land.
    """
    assert_networkspec_invariants(root)
    seen: set[int] = set()
    yield from _walk_networks(root, owner_path, seen)


def _walk_networks(obj: Any, path: tuple[str, ...],
                   seen: set[int]) -> Iterator[tuple[NetworkSpec, 'OwnerContext']]:
    if not isinstance(obj, BaseModel):
        return
    if id(obj) in seen:
        return
    seen.add(id(obj))

    cls = type(obj)
    for fname in cls.__pydantic_fields__:
        value = getattr(obj, fname)
        yield from _yield_from_value(value, (*path, fname), seen, owner=obj)


def _yield_from_value(value: Any, field_path: tuple[str, ...],
                      seen: set[int],
                      owner: Optional[BaseModel] = None
                      ) -> Iterator[tuple[NetworkSpec, 'OwnerContext']]:
    if isinstance(value, NetworkSpec):
        parent_path = field_path[:-1]
        ctx = OwnerContext(owner_path=parent_path, owner=owner)
        yield value, ctx
        return
    if isinstance(value, BaseModel):
        yield from _walk_networks(value, field_path, seen)
        return
    if isinstance(value, (list, tuple)):
        for i, elt in enumerate(value):
            yield from _yield_from_value(elt, (*field_path, str(i)), seen, owner=owner)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _yield_from_value(v, (*field_path, str(k)), seen, owner=owner)



# ---------------------------------------------------------------------------
# Task 039 N3: discriminated-union TypeAdapters used by abstract-base
# ``model_validate`` overrides and by system-container fields.
# ---------------------------------------------------------------------------

_BladeFabricAdapter: TypeAdapter[Any] = TypeAdapter(_BladeFabricUnion)
_RackFabricAdapter: TypeAdapter[Any] = TypeAdapter(_RackFabricUnion)
_DataCenterFabricAdapter: TypeAdapter[Any] = TypeAdapter(_DataCenterFabricUnion)
