
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field

_registry_lock = threading.Lock()

@dataclass(frozen=True)
class OpRegistryEntry:
    opname         : str
    group          : str
    min_input      : int
    max_input      : int
    min_output     : int
    max_output     : int
    shape_inf_func : Callable
    attrs          : frozenset[str]    = field(default_factory=frozenset)
    aliases        : Mapping[str, str] = field(default_factory=dict)

_STANDARD_DOMAIN = ''

class TensorOpRegistry:
    def __init__(self):
        self._registry : dict[str, OpRegistryEntry]= {}
        #group -> (domain string, opset version)
        self._group_domains: dict[str, tuple] = {}

    def register(self, entry: OpRegistryEntry) -> None:
        for alias_key, cannon_key in entry.aliases.items():
            if cannon_key not in entry.attrs:
                raise ValueError(
                        f"register_ops({entry.opname}): alias {alias_key!r}->{cannon_key!r} "
                        f"targets key {cannon_key!r} not in declared attrs {set(entry.attrs)!r}"
                        )

            if alias_key in entry.attrs:
                raise ValueError(
                        f"register_ops({entry.opname}): alias {alias_key!r} collides "
                        f"with cannonical attr of the same name"
                        )

        if entry.opname in self._registry:
            raise ValueError(
                        f"register_ops({entry.opname}): opname already registered "
                        f"by group={self._registry[entry.opname].group!r} "
                        f"choose a unique optype to avoid silent override"
                    )

        self._registry[entry.opname] = entry


    def get_op(self, opname: str) -> OpRegistryEntry:
        try:
            return self._registry[opname]
        except KeyError:
            raise KeyError(f"{opname} not supported in TensorOpRegistry!!")

    def get_shape_inference_function(self, opname) -> Callable:
        assert opname in self._registry, f"operator= {opname} is not registered with TensorOpRegistry"
        return self._registry[opname].shape_inf_func


    # -- custom domain management --
    def register_domain(self, group: str, domain: str, *,
                        opset_version: int = 1) -> None:
        if domain == _STANDARD_DOMAIN:
            return

        existing = self._group_domains.get(group)
        if existing is not None and existing != (domain, opset_version):
            raise ValueError(
                    f"register_domain({group!r}): already registered as {existing!r}, "
                    f"cannot rebind to ({domain!r}, {opset_version})"
                    )
        self._group_domains[group] = (domain, opset_version)
        return

    def get_op_domain(self, opname: str) -> str:
        rec = self._registry.get(opname)
        if rec is None:
            return _STANDARD_DOMAIN
        dom = self._group_domains.get(rec.group)
        return dom[0] if dom is not None else _STANDARD_DOMAIN

    def is_custom_op(self, opname: str) -> bool:
        return self.get_op_domain(opname) != _STANDARD_DOMAIN

    def custom_domains_for(self, opnames: Iterable[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in opnames:
            rec = self._registry.get(n)
            if rec is None:
                continue
            dom = self._group_domains.get(rec.group)
            if dom is None:
                continue
            d_name, d_ver = dom
            prev = out.get(d_name)
            if prev is None or d_ver > prev:
                out[d_name] = d_ver
        return out



# Global registry instance
_global_registry: TensorOpRegistry | None = None

def get_op_registry() -> TensorOpRegistry:
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = TensorOpRegistry()
    return _global_registry

def register_ops(group, optbl):
    for rec in optbl:
        if len(rec) == 6:
            opname, max_i, min_i, max_o, min_o, sinf = rec
            attrs_i, aliases_i = {}, {} #type: ignore
        elif len(rec) == 7:
            opname, max_i, min_i, max_o, min_o, sinf, attrs_i = rec
            aliases_i = {} #type: ignore
        elif len(rec) == 8:
            opname, max_i, min_i, max_o, min_o, sinf, attrs_i, aliases_i = rec
        else:
            raise ValueError(
                    f"register_ops: unsupported row arity {len(rec)} for {rec[0]!r}"
                    )
        entry = OpRegistryEntry(
                opname = opname,
                group = group,
                min_input = min_i,
                max_input = max_i,
                min_output = min_o,
                max_output = max_o,
                shape_inf_func = sinf,
                attrs = frozenset(attrs_i),
                aliases = dict(aliases_i),
                )
        get_op_registry().register(entry)


AISLE_DOMAIN = 'com.aisle'

def register_domain(group: str, domain: str, *, opset_version: int = 1) -> None:
    get_op_registry().register_domain(group, domain, opset_version=opset_version)

def get_op_domain(opname: str) -> str:
    return get_op_registry().get_op_domain(opname)

def is_custom_op(opname: str) -> bool:
    return get_op_registry().is_custom_op(opname)

def custom_domains_for(opnames: Iterator[str]):
    return get_op_registry().custom_domains_for(opnames)
