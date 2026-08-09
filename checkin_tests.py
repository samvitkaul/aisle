
import argparse
import os
import re
import subprocess
import sys
from collections.abc import Callable
from shutil import rmtree
from typing import Any

ODIR = '__RUN_TESTS'
LOGD = f'{ODIR}/logs'
OptionalString = str | None

CommandHandler = Callable[[OptionalString], list[str]]

def prepare_coverage(condaprefix: OptionalString) -> list[str]:
    commands = [
            f'{condaprefix} coverage erase && coverage run -m pytest --durations=0 -m "not slow and not integration" && coverage combine && coverage report && coverage html',
            ]
    return commands

def prepare_unit_slow(condaprefix: OptionalString) -> list[str]:
    # roughly >= 0.5s
    commands = [
            f'{condaprefix} python -m pytest --durations=0 -m "slow and not integration"'
            ]
    return commands

def prepare_static(condaprefix: OptionalString) -> list[str]:
    commands = [
            f'{condaprefix} ruff check src/ test/ workloads/ checkin_tests.py',
            f'{condaprefix} mypy ./',
            ]
    return commands



def run_a_job(cmd: str, ofilename: str) -> subprocess.CompletedProcess:
    with open(ofilename, 'w') as OUT:
        print('Command:', cmd, file=OUT)
        print(file=OUT)
        OUT.flush()
        return subprocess.run(cmd, shell=True, executable='/bin/bash', stdout=OUT, stderr=subprocess.STDOUT, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', '-e', dest='condaenv', help='Conda env name or path to be used for tests')
    parser.add_argument('--stop',        '-x', action='store_true', help='Stop tests on first failure')
    parser.add_argument('--dryrun',      '-n', action='store_true', help='Show but do not execute commands')
    parser.add_argument('--filter',      help='filter commands')
    parser.add_argument('--tests', nargs='*', choices = ['coverage',
                                                         #'unit_slow',
                                                         'static', 'all'])
    args = parser.parse_args()
    test_handlers: dict[str, CommandHandler] = {
            'coverage': prepare_coverage,
            #'unit_slow': prepare_unit_slow,
            'static': prepare_static,
    }
    enabled_tests: set = set()
    if not args.tests or 'all' in args.tests:
        enabled_tests = {x for x in test_handlers}
    else:
        enabled_tests = set(args.tests)
        print(">>", enabled_tests)
        unsupported_tests = enabled_tests - set(test_handlers.keys())
        print(">>", unsupported_tests)
        if unsupported_tests:
            print(f'error: tests {unsupported_tests} are not supported!!')
            sys.exit(1)

    condaenvprefix: OptionalString = ''
    if args.condaenv is not None:
        condabase: str = os.popen('conda info --base').read().strip()
        condaenvprefix = f'source {condabase}/etc/profile.d/conda.sh && conda activate {args.condaenv} && '

    commands = []
    for test in sorted(enabled_tests):
        commands.extend(test_handlers[test](condaenvprefix))

    if args.filter:
        commands = [cmd for cmd in commands if re.search(args.filter, cmd)]

    if not args.dryrun:
        rmtree(ODIR, ignore_errors=True)
        os.makedirs(ODIR, exist_ok=True)
        os.makedirs(LOGD, exist_ok=True)

    num_failures: int = 0
    results: list[dict[str, Any]] = [{} for _ in range(len(commands))]
    for cmdno, cmd in enumerate(commands):
        cmd = cmd.replace('--study PLACEHOLDER ', f'--study study_{cmdno+1:03} ')
        study_match = re.search('--study', cmd)
        if study_match:
            log_file = os.path.join(LOGD, f'study_{cmdno+1:03}.log')
        else:
            log_file = os.path.join(LOGD, f'checkin_test_{cmdno+1:03}.log')
        print(f'#{cmdno+1}/{len(commands)}: {cmd} -> {log_file}')

        if args.dryrun:
            continue

        cmdret = run_a_job(cmd, log_file)
        results[cmdno] = {
                'id': f'{cmdno:2d}-{log_file}',
                'returncode': cmdret.returncode,
                'status': 'PASS' if cmdret.returncode == 0 else 'FAIL',
                }
        if cmdret.returncode != 0:
            if args.stop:
                print(f'error: {cmd} failed with exit code {cmdret.returncode}')
                print('checkin tests failed')
                return cmdret.returncode
            num_failures += 1

    if args.dryrun:
        return 0

    mf = max(len(x['id']) for x in results)
    for res in ['PASS', 'FAIL']:
        if res == 'FAIL' and num_failures:
            print('-- Failed commands')
        for cmdno, result in enumerate(results):
            if result['status'] != res:
                continue
            print(f'{result["id"]:{mf}s} RESULT= {res}')

    if num_failures == 0:
        errorlines = os.popen(f'grep ERROR: {LOGD}/*.log').readlines()
        if errorlines:
            print('Warning: log lines contain ERROR msgs')
            for line in errorlines:
                line = line.strip()
                print('\t' + line)
            print('----------------------------------------------------------------------------')
            print('Warning: log lines contain ERROR msgs though runs with exit code 0')
            print('----------------------------------------------------------------------------')

    if num_failures:
        print(f'{num_failures} of {len(commands)} failed')
    else:
        print('checkin tests successful')
    return num_failures


if __name__ == '__main__':
    sys.exit(main())
