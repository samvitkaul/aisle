
from typing import Any

from pydantic import BaseModel, Field

from src.utils.common import parse_yaml

type OpName   = str
type PipeName = str

class OpRemovalSpec(BaseModel, extra='forbid', populate_by_name=False, frozen=True):
    op_names: set[OpName]

    def check(self, op_name: OpName) -> bool:
        return op_name in self.op_names

    @staticmethod
    def from_list(op_list: list[OpName]) -> 'OpRemovalSpec':
        return OpRemovalSpec(op_names={x.upper() for x in (op_list or [])})

class OpFusionSpec(BaseModel):
    op_sequences: list[list[OpName]] = Field(..., description="Prioritized Op Fusion Sequences")

    def get_op_fusion_sequence(self):
        for seq in self.op_sequences:
            yield from seq

    @staticmethod
    def from_list(spec: list[list[OpName]]) -> 'OpFusionSpec':
        op_fusion_list = [[y.upper() for y in x] for x in (spec or [])]
        return OpFusionSpec(op_sequences=op_fusion_list)

class ResourceMap(BaseModel):
    op_map: dict[OpName, PipeName] = Field(..., description='op2rsrc mapping')

    def op2pipe(self, op_name: OpName) -> PipeName:
        try:
            pipe = self.op_map[op_name.upper()]
        except KeyError:
            raise KeyError(f'Op {op_name} not found in op2rsrc mapping')

        return pipe

    @staticmethod
    def from_dict(spec: dict[str, Any]) -> 'ResourceMap':
        if not spec:
            return ResourceMap(op_map={})
        if 'compute' not in spec:
            raise KeyError('Attribute(compute) missing in op2rsrc spec')
        tbl = {}
        for op_pipe, op_list in spec['compute'].items():
            tbl.update({o.upper(): op_pipe.lower() for o in op_list})
        return ResourceMap(op_map=tbl)

class MapInfo(BaseModel, extra='forbid', frozen=True):
    op_removal_spec: OpRemovalSpec | None = None
    op_fusion_spec : OpFusionSpec | None  = None
    rsrc_spec      : ResourceMap | None   = None

    @staticmethod
    def from_yaml(cfg_yaml_file: str) -> 'MapInfo':
        cfg_dict = parse_yaml(cfg_yaml_file) or {}

        # No required fields today. Add to this list to enforce presence later.
        required_fields: list[str] = []
        for ff in required_fields:
            if ff not in cfg_dict:
                raise KeyError(f'required attribute: {ff} missing in mapping file: {cfg_yaml_file}')

        kwargs: dict[str, Any] = {}
        if cfg_dict.get('op_removal_spec'):
            kwargs['op_removal_spec'] = OpRemovalSpec.from_list(cfg_dict['op_removal_spec'])
        if cfg_dict.get('op_fusion_spec'):
            kwargs['op_fusion_spec']  = OpFusionSpec.from_list(cfg_dict['op_fusion_spec'])
        if cfg_dict.get('rsrc_spec'):
            kwargs['rsrc_spec']       = ResourceMap.from_dict(cfg_dict['rsrc_spec'])

        return MapInfo(**kwargs)
