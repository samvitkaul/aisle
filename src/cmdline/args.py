
import argparse


def setup_cmdline_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser('aisle')
    parser.add_argument('--odir',    '-o',  required=True)
    parser.add_argument('--study',   '-s',  required=True)
    parser.add_argument('--wlspec',  '-w',  required=True)
    parser.add_argument('--devspec', '-ds', required=True)
    parser.add_argument('--devtype', '-dt', required=True)
    parser.add_argument('--mapspec', '-p',  required=True)
    parser.add_argument('--dryrun',  '-n', action='store_true', help='Show but do not execute commands')
    args = parser.parse_args()
    return args
