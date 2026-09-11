
from .base import GraphPass as GraphPass, PassConfig as PassConfig, PassPipeline as PassPipeline

def default_pipeline() -> PassPipeline:
    from .op_removal import OpRemovalPass
    from .constant_folding import ConstantFoldingPass
    from .dead_node_elimination import DeadNodeEliminationPass
    from .resource_mapping import ResourceMappingPass
    from .op_fusion import OpFusionPass

    pipeline = PassPipeline()
    pipeline.add(OpRemovalPass())
    pipeline.add(ConstantFoldingPass())
    pipeline.add(DeadNodeEliminationPass())
    pipeline.add(ResourceMappingPass())
    pipeline.add(OpFusionPass())
    return pipeline
