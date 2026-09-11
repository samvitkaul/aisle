
from .base import GraphPass as GraphPass
from .base import PassConfig as PassConfig
from .base import PassPipeline as PassPipeline


def default_pipeline() -> PassPipeline:
    from .constant_folding import ConstantFoldingPass
    from .dead_node_elimination import DeadNodeEliminationPass
    from .op_fusion import OpFusionPass
    from .op_removal import OpRemovalPass
    from .resource_mapping import ResourceMappingPass

    pipeline = PassPipeline()
    pipeline.add(OpRemovalPass())
    pipeline.add(ConstantFoldingPass())
    pipeline.add(DeadNodeEliminationPass())
    pipeline.add(ResourceMappingPass())
    pipeline.add(OpFusionPass())
    return pipeline
