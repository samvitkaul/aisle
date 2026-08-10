
from src.utils.profiling import bracket
from src.cmdline import setup_cmdline_args
from src.config import WLInfo


from loguru import logger
import time
import argparse

INFO    = logger.info
DEBUG   = logger.debug
WARNING = logger.warning
ERROR   = logger.error


def simulation(args: argparse.Namespace) -> tuple[int, int]:
    return 0,0

def main(argv: list[str] | None = None) -> tuple[int, int]:
    args = setup_cmdline_args(argv)
    return simulation(args)

if __name__ == '__main__':
    start_time = time.perf_counter()
    exp_count, exp_run = main()
    end_time = time.perf_counter()
    del_time = end_time - start_time
    infostr = f"Completed {exp_run}/{exp_count} experiments in {del_time:0.2f} secs"
    if exp_run > 0:
        runs_per_sec = exp_run / del_time
        infostr += f" @ {runs_per_sec: 0.2f} runs/sec"
    INFO(infostr)
