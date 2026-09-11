

from pydantic import BaseModel, Field

from .knob import KnobVal


class ExecOpStats(BaseModel, extra='forbid', populate_by_name=False, frozen=True):
    wltype           : str
    wlname           : str
    wlbasedir        : str
    wlmodule         : str
    wli_name         : str
    batchsize        : int
    system_type      : str
    system           : str
    clocks           : dict[str, KnobVal]
    tick_freq_Mhz    : int
    ticks_per_cycle  : dict[str, int]
    precision        : str
    opnum            : int
    opname           : str
    is_input_node    : bool
    is_output_node   : bool
    optype           : str
    op_rpt_count     : int
    attrs            : dict
    inList           : list
    outList          : list
    removed          : bool
    removal_reason   : str
    fused            : bool
    fused_with_op    : str
    inBytes          : int
    outBytes         : int
    instrs           : dict
    inParamCount     : int
    inActCount       : int
    outActCount      : int
    instr_count      : int
    compute_ticks    : dict[str, int]
    mem_ticks        : dict[str, dict[str, int]]
    compute_util     : dict[str, float]
    mem_util         : dict[str, dict[str, float]]
    ideal_ticks      : int
    ideal_msecs      : float
    ticks            : int
    msecs            : float
    rsrc_bnck        : str
    network_ticks       : dict[str, dict[str, int]]
    network_ticks_total : int
    network_msecs    : float
    unpriced_reason  : str = ''

class SummaryStatsRecord(BaseModel, extra='forbid', populate_by_name=False, frozen=True):
    wltype                   : str
    wlname                   : str
    wlbasedir                : str
    wlmodule                 : str
    wli_name                 : str
    batchsize                : int
    dp                       : int = 1
    tp                       : int = 1
    ep                       : int = 1
    pp                       : int = 1
    world_size               : int = 1
    system_type              : str
    system                   : str
    clocks                   : dict[str, KnobVal]
    tick_freq_Mhz            : int
    ticks_per_cycle          : dict[str, int]
    mem_size_GB              : float
    mem_size_per_rank_GB     : float = 0.0
    device_memsize_GB        : float
    fits_device              : bool
    #device_peak_bw_GBps      : float
    inParams                 : int
    inActs                   : int
    outActs                  : int
    maxActs                  : int
    inParamBytes             : int
    inActBytes               : int
    outActBytes              : int
    maxActBytes              : int
    tot_ideal_ticks          : int
    tot_ideal_msecs          : float
    tot_ticks                : int
    tot_msecs                : float
    ideal_throughput         : float
    perf_projection          : float
    tot_compute_ticks        : int
    tot_mem_rd_ticks         : int
    tot_mem_wr_ticks         : int
    tot_compute_util         : float
    tot_mem_rd_util          : float
    tot_mem_wr_util          : float
    tot_network_util         : float = 0.0
    rsrc_comp                : float
    rsrc_mem                 : float
    rsrc_nw                  : float = 0.0
    tot_network_ticks        : int
    tot_network_msecs        : float
    network_ticks_by_network : dict[str, int]
    network_util_by_network  : dict[str, float] = Field(default_factory=dict)
    n_ccl_ops                : int
    n_priced_ccl_ops         : int = 0
    n_unpriced_ccl_ops       : int = 0

class ExecStatsReport(BaseModel, extra='forbid', populate_by_name=False, frozen=True):
    opstats: list[ExecOpStats]

class SummaryStats(BaseModel, extra='forbid', populate_by_name=False, frozen=True):
    runstats: list[SummaryStatsRecord]


_EXECOPSTATS_FIELDS: tuple[str, ...] = tuple(ExecOpStats.model_fields)
_SUMMARYSTATS_FIELDS: tuple[str, ...] = tuple(SummaryStatsRecord.model_fields)


def opstats_to_row(r: ExecOpStats) -> tuple:
    """Tuple-emitting fast path for CSV. Replaces r.model_dump() +
    DictWriter dict-lookup. Field order matches _EXECOPSTATS_FIELDS."""
    clocks_str = {k: v.serialize() for k, v in r.clocks.items()}
    return (
        r.wltype, r.wlname, r.wlbasedir, r.wlmodule, r.wli_name,
        r.batchsize, r.system_type, r.system, clocks_str,
        r.tick_freq_Mhz, r.ticks_per_cycle, r.precision,
        r.opnum, r.opname, r.is_input_node, r.is_output_node,
        r.optype, r.op_rpt_count, r.attrs, r.inList, r.outList,
        r.removed, r.removal_reason, r.fused, r.fused_with_op,
        r.inBytes, r.outBytes, r.instrs,
        r.inParamCount, r.inActCount, r.outActCount, r.instr_count,
        r.compute_ticks, r.mem_ticks, r.compute_util, r.mem_util,
        r.ideal_ticks, r.ideal_msecs, r.ticks, r.msecs,
        r.rsrc_bnck, r.network_ticks, r.network_ticks_total,
        r.network_msecs, r.unpriced_reason,
    )


def summary_to_row(r: SummaryStatsRecord) -> tuple:
    """Tuple-emitting fast path for summary CSV. Field order matches
    _SUMMARYSTATS_FIELDS."""
    clocks_str = {k: v.serialize() for k, v in r.clocks.items()}
    return tuple(
        clocks_str if fname == 'clocks' else getattr(r, fname)
        for fname in _SUMMARYSTATS_FIELDS
    )

