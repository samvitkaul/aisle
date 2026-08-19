import pytest

from src.config.device import Instruction, ComputeCore, Cache, Memory, GPUDie, GPU
from src.config.system.blade import Blade
from src.config.system.rack import Rack
from src.config.system.cluster import Cluster

from src.config.parser import parse_config_with_refs
from src.config.workload import WLInfo
#from src.front import AxisSpec, ParallelScheme, SymbolicParallelSpec

from src.utils.data_types import DataType

from pydantic import ValidationError
from loguru import logger

INFO = logger.info


@pytest.mark.unit
@pytest.mark.parametrize('idx,cfg_yaml_file', enumerate(['config/tests/instructions.yml']))
def test_instructions(idx, cfg_yaml_file):
    idb = parse_config_with_refs(cfg_yaml_file, inject_names=True, ignore_keys=['opc', 'throughput'])

    try:
        for isa_name, isa_spec in idb.items():
            # isa_spec now has 'name' field (injected), plus instruction definitions
            assert isa_spec.get('name') == isa_name, f'ISA name mismatch: {isa_name}'

            # Iterate through instructions (skip the 'name' field)
            for instr_name, instr_data in isa_spec.items():
                if instr_name == 'name':
                    continue

                instr_spec = Instruction.model_validate(instr_data)

                assert instr_spec.name == instr_name, f'Instruction name mismatch: {instr_name}'
                INFO(f'Instruction: {instr_spec.name:20} -> {instr_spec}')

                # Test precision lookups
                for dt in DataType:
                    dt_str = f'{dt}'
                    if instr_spec.has_precision(dt_str):
                        tpt = instr_spec.get_throughput(dt_str)
                        INFO(f'  {dt_str:8}: {tpt}')

    except ValidationError as e:
        print(e)
        raise


@pytest.mark.unit
@pytest.mark.parametrize('idx,cfg_yaml_file', enumerate(['config/tests/gpu_cores.yml']))
def test_gpu_cores(idx, cfg_yaml_file):

    cores_db = parse_config_with_refs(cfg_yaml_file, inject_names=True, ignore_keys=['opc'])

    try:
        for core_name, core_spec in cores_db.items():
            # core_spec now has 'name' (injected), 'frequency', and 'instructions'
            cspec = ComputeCore.model_validate(core_spec)

            assert cspec.name == core_name, f'Core name mismatch: {core_name}'
            INFO(f'\nComputeCore: {cspec.name}')
            INFO(f'  Frequency: {cspec.frequency}')
            INFO(f'  Instructions: {sorted(cspec.instructions.keys())}')

            # Test specific instructions based on core type
            if core_name == 'tensor_core':
                mac_instr = cspec.get_instr('mac')
                mac_opc = cspec.peak_ops_per_cycle('mac', 'fp8')
                mac_gflops = cspec.peak_flops('mac', 'fp8', units='GFLOPS', mul_factor=2)
                INFO(f'  MAC: {mac_instr.name}, OPS/cycle={mac_opc}, GFLOPS={mac_gflops:.1f}')
            else:  # cuda_core
                mul_instr = cspec.get_instr('mul')
                mul_opc = cspec.peak_ops_per_cycle('mul', 'int32')
                mul_mflops = cspec.peak_flops('mul', 'fp32', units='MFLOPS')
                erf_instr = cspec.get_instr('erf')
                erf_opc = cspec.peak_ops_per_cycle('erf', 'int32')
                erf_mflops = cspec.peak_flops('erf', 'fp32', units='MFLOPS')
                INFO(f'  MUL: {mul_instr.name}, OPS/cycle={mul_opc}, MFLOPS={mul_mflops:.1f}')
                INFO(f'  ERF: {erf_instr.name}, OPS/cycle={erf_opc}, MFLOPS={erf_mflops:.1f}')

    except ValidationError as e:
        print(e)
        raise


@pytest.mark.unit
@pytest.mark.parametrize('idx,cfg_yaml_file', enumerate(['config/tests/cache.yml']))
def test_caches(idx, cfg_yaml_file):
    caches = parse_config_with_refs(cfg_yaml_file, inject_names=True, ignore_keys=['opc'])
    try:
        for cname, cinfo in caches.items():
            cspec = Cache.model_validate(cinfo)
            INFO('CACHE', cspec)
    except ValidationError as e:
        print(e)
        raise
    return


@pytest.mark.unit
@pytest.mark.parametrize('idx,cfg_yaml_file', enumerate(['config/tests/memory.yml']))
def test_memories(idx, cfg_yaml_file):
    memories = parse_config_with_refs(cfg_yaml_file, inject_names=True, ignore_keys=['opc'])
    try:
        for k, v in memories.items():
            mem_spec = Memory.model_validate(v)
            sz = mem_spec.get_size(units='MB')
            fq = mem_spec.get_frequency(units='GHz')
            bpc = mem_spec.peak_bytes_per_cycle()
            bw = mem_spec.peak_bandwidth(units='THz')
            INFO(f'MEM: size={sz}, frequency={fq}, bpc={bpc}, bw={bw}')
    except ValidationError as e:
        print(e)
        raise
    return


@pytest.mark.unit
@pytest.mark.parametrize('idx,cfg_yaml_file', enumerate(['config/tests/gpu_dies.yml']))
def test_gpu_dies(idx, cfg_yaml_file):
    db = parse_config_with_refs(cfg_yaml_file, inject_names=True, ignore_keys=['opc'])
    try:
        for gpu_die_name, gpu_die_info in db.items():
            spec = GPUDie.model_validate(gpu_die_info)
            print(spec)
    except ValidationError as e:
        print(e)
        raise

    return


@pytest.mark.unit
@pytest.mark.parametrize('idx,cfg_yaml_file', enumerate(['config/tests/gpus.yml']))
def test_gpus(idx, cfg_yaml_file):
    db = parse_config_with_refs(cfg_yaml_file, inject_names=True, ignore_keys=['opc'])
    try:
        for gpu_name, gpu_info in db.items():
            spec = GPU.model_validate(gpu_info)
            print(spec)
    except ValidationError as e:
        print(e)
        raise
    return


@pytest.mark.unit
@pytest.mark.parametrize('idx,cfg_yaml_file', enumerate(['config/tests/blades.yml']))
def test_blades(idx, cfg_yaml_file):
    db = parse_config_with_refs(cfg_yaml_file, inject_names=True, ignore_keys=['opc'])
    try:
        for blade_name, blade_info in db.items():
            spec = Blade.model_validate(blade_info)
            print(spec)
    except ValidationError as e:
        print(e)
        raise
    return


@pytest.mark.unit
@pytest.mark.parametrize('idx,cfg_yaml_file', enumerate(['config/tests/racks.yml']))
def test_racks(idx, cfg_yaml_file):
    db = parse_config_with_refs(cfg_yaml_file, inject_names=True, ignore_keys=['opc'])
    try:
        for rack_name, rack_info in db.items():
            spec = Rack.model_validate(rack_info)
            print(spec)
    except ValidationError as e:
        print(e)
        raise
    return


@pytest.mark.unit
@pytest.mark.parametrize('idx,cfg_yaml_file', enumerate(['config/tests/clusters.yml']))
def test_clusters(idx, cfg_yaml_file):
    db = parse_config_with_refs(cfg_yaml_file, inject_names=True, ignore_keys=['opc'])
    try:
        for cluster_name, cluster_info in db.items():
            spec = Cluster.model_validate(cluster_info)
            print(spec)
    except ValidationError as e:
        print(e)
        raise
    return


# ===================================================================
# ComputeCore unit tests
# ===================================================================


def _make_compute_core(**overrides):
    """Build a minimal ComputeCore for unit testing."""
    base = {
        'name': 'test_core',
        'frequency': '1.0 GHz',
        'instructions': {
            'mac': {'name': 'mac', 'opc': {'fp32': 256.0, 'fp16': 512.0}},
            'mul': {'name': 'mul', 'opc': {'fp32': 128.0, 'int32': 64.0}},
        },
    }
    base.update(overrides)
    return ComputeCore.model_validate(base)


class TestComputeCoreUnit:
    @pytest.mark.unit
    def test_coerce_instructions_non_mapping_raises(self):
        with pytest.raises((ValidationError, TypeError)):
            ComputeCore.model_validate({'name': 'bad', 'frequency': '1.0 GHz', 'instructions': 'not_a_mapping'})

    @pytest.mark.unit
    def test_get_instr_missing_raises(self):
        core = _make_compute_core()
        with pytest.raises(KeyError, match='not found'):
            core.get_instr('nonexistent')

    @pytest.mark.unit
    def test_handle_missing_precision_no_fallback_raises(self):
        core = _make_compute_core()
        instr = core.get_instr('mac')
        with pytest.raises(ValueError, match='Missing precision'):
            core.handle_missing_precision(instr, 'int64')

    @pytest.mark.unit
    def test_peak_ops_per_cycle_strict_missing(self):
        """strict=True with missing precision returns 0.0."""
        core = _make_compute_core()
        result = core.peak_ops_per_cycle('mac', 'int64', strict=True)
        assert result == 0.0

    @pytest.mark.unit
    def test_peak_ops_per_cycle_fallback(self):
        """Without strict, missing precision falls back to compatible type."""
        core = _make_compute_core()
        # fp8 should fall back to fp16 or fp32
        result = core.peak_ops_per_cycle('mac', 'fp8')
        assert result > 0

    @pytest.mark.unit
    def test_peak_flops_tflops(self):
        core = _make_compute_core()
        result = core.peak_flops('mac', 'fp32', units='TFLOPS')
        # 256 ops/cycle * 1e9 Hz = 256e9 FLOPS = 0.256 TFLOPS
        assert abs(result - 0.256) < 0.01

    @pytest.mark.unit
    def test_peak_flops_gflops(self):
        core = _make_compute_core()
        result = core.peak_flops('mac', 'fp32', units='GFLOPS')
        # 256 ops/cycle * 1e9 Hz = 256e9 FLOPS = 256 GFLOPS
        assert abs(result - 256.0) < 0.1


class TestWLInfoParallelSpecShim:
    @pytest.mark.unit
    def notest_property_shim_returns_concrete(self):
        wl = WLInfo(
            wltype='BTEN',
            wlname='w',
            basedir='workloads',
            source='DistributedMLP.py',
            wli_name='inst',
            wli_params={},
            parallel_spec=ParallelScheme(tp=4),
        )
        assert wl.parallel_scheme == ParallelScheme(tp=4)

    @pytest.mark.unit
    def notest_property_shim_returns_none_for_symbolic(self):
        sym = SymbolicParallelSpec(axes={'tp': AxisSpec(domain='choices', choices=(2, 4))}, constraints=(), defaults={})
        wl = WLInfo(
            wltype='BTEN',
            wlname='w',
            basedir='workloads',
            source='DistributedMLP.py',
            wli_name='inst',
            wli_params={},
            parallel_spec=sym,
        )
        assert wl.parallel_scheme is None

    @pytest.mark.unit
    def notest_with_parallel_scheme_wrapper_unchanged(self):
        wl = WLInfo(
            wltype='BTEN', wlname='w', basedir='workloads', source='DistributedMLP.py', wli_name='inst', wli_params={}
        )
        updated = wl.with_parallel_scheme(ParallelScheme(dp=2, tp=4))
        assert updated.parallel_spec == ParallelScheme(dp=2, tp=4)
        assert updated.parallel_scheme == ParallelScheme(dp=2, tp=4)


# ===================================================================
# Mapping error path tests
# ===================================================================

from src.config.mapping import ResourceMap


class TestMappingErrorPaths:
    @pytest.mark.unit
    def test_op2pipe_missing_op_raises(self):
        rm = ResourceMap(op_map={'MATMUL': 'tensor_core'})
        with pytest.raises(KeyError, match='not found'):
            rm.op2pipe('NonExistent')

    @pytest.mark.unit
    def test_from_dict_empty_returns_empty_map(self):
        """Task 013: empty dict is now permitted and yields an empty ResourceMap."""
        rm = ResourceMap.from_dict({})
        assert rm.op_map == {}

    @pytest.mark.unit
    def test_from_dict_missing_compute_raises(self):
        with pytest.raises(KeyError, match='compute'):
            ResourceMap.from_dict({'other_key': {}})


# ===================================================================
# Task 013: MapInfo Optional Fields
# ===================================================================

from src.config.mapping import MapInfo, OpRemovalSpec, OpFusionSpec


class TestMapInfoOptionalFields:
    @pytest.mark.unit
    def test_mapinfo_construct_default(self):
        """MapInfo() constructs with all three fields defaulting to None."""
        m = MapInfo()
        assert m.op_removal_spec is None
        assert m.op_fusion_spec is None
        assert m.rsrc_spec is None

    @pytest.mark.unit
    def test_mapinfo_from_yaml_all_three_present(self):
        """Regression: config/mappings.yml continues to load all three sections."""
        m = MapInfo.from_yaml('config/mappings.yml')
        assert m.op_removal_spec is not None
        assert m.op_fusion_spec is not None
        assert m.rsrc_spec is not None
        # Sanity-check contents
        assert 'IDENTITY' in m.op_removal_spec.op_names
        assert 'CONSTANT' in m.op_removal_spec.op_names
        assert any('MATMUL' in seq for seq in m.op_fusion_spec.op_sequences)
        assert m.rsrc_spec.op_map.get('MATMUL') == 'tensor'

    @pytest.mark.unit
    def test_mapinfo_from_yaml_missing_fusion(self, tmp_path):
        yml = tmp_path / 'm.yml'
        yml.write_text('op_removal_spec:\n  - Identity\nrsrc_spec:\n  compute:\n    tensor: [Matmul]\n')
        m = MapInfo.from_yaml(str(yml))
        assert m.op_removal_spec is not None
        assert m.op_fusion_spec is None
        assert m.rsrc_spec is not None

    @pytest.mark.unit
    def test_mapinfo_from_yaml_missing_rsrc(self, tmp_path):
        yml = tmp_path / 'm.yml'
        yml.write_text('op_removal_spec:\n  - Identity\nop_fusion_spec:\n  - [Matmul, Add]\n')
        m = MapInfo.from_yaml(str(yml))
        assert m.op_removal_spec is not None
        assert m.op_fusion_spec is not None
        assert m.rsrc_spec is None

    @pytest.mark.unit
    def test_mapinfo_from_yaml_missing_removal(self, tmp_path):
        yml = tmp_path / 'm.yml'
        yml.write_text('op_fusion_spec:\n  - [Matmul, Add]\nrsrc_spec:\n  compute:\n    tensor: [Matmul]\n')
        m = MapInfo.from_yaml(str(yml))
        assert m.op_removal_spec is None
        assert m.op_fusion_spec is not None
        assert m.rsrc_spec is not None

    @pytest.mark.unit
    def test_mapinfo_from_yaml_all_missing(self, tmp_path):
        """Empty / comment-only YAML yields a MapInfo() with all three fields None."""
        yml = tmp_path / 'm.yml'
        yml.write_text('# only a comment\n')
        m = MapInfo.from_yaml(str(yml))
        assert m.op_removal_spec is None
        assert m.op_fusion_spec is None
        assert m.rsrc_spec is None

        yml2 = tmp_path / 'empty.yml'
        yml2.write_text('')
        m2 = MapInfo.from_yaml(str(yml2))
        assert m2.op_removal_spec is None
        assert m2.op_fusion_spec is None
        assert m2.rsrc_spec is None

    @pytest.mark.unit
    def test_mapinfo_from_yaml_empty_lists(self, tmp_path):
        """Empty-list / empty-dict values collapse to None (absent OR empty rule)."""
        yml = tmp_path / 'm.yml'
        yml.write_text('op_removal_spec: []\nop_fusion_spec: []\nrsrc_spec: {}\n')
        m = MapInfo.from_yaml(str(yml))
        assert m.op_removal_spec is None
        assert m.op_fusion_spec is None
        assert m.rsrc_spec is None

    @pytest.mark.unit
    def test_op_removal_spec_from_empty_list_ok(self):
        """OpRemovalSpec.from_list([]) no longer raises (defensive hardening)."""
        s = OpRemovalSpec.from_list([])
        assert s.op_names == set()

    @pytest.mark.unit
    def test_op_fusion_spec_from_empty_list_ok(self):
        """OpFusionSpec.from_list([]) no longer raises (defensive hardening)."""
        s = OpFusionSpec.from_list([])
        assert s.op_sequences == []
