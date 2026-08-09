
import csv
import json
from collections.abc import KeysView
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

import yaml


def prod_ints(L: list[int]) -> int:
    return reduce(mul, L, 1)

def str_to_bool(s):
    if isinstance(s, bool):
        return s
    if isinstance(s, int):
        return s != 0
    if isinstance(s, float):
        return s != 0
    if s.lower() in ['true', 't', 'yes', 'y', 'on', 'enable', '1']:
        return True
    elif s.lower() in ['false', 'f', 'no', 'n', 'off', 'disable', '0']:
        return False
    else:
        raise ValueError('expecting boolean value')

def parse_csv(csvfilename):
    with open(csvfilename, 'r') as fcsv:
        rowlines = [row.strip() for row in fcsv]

    # Skip rows beginning with '#', and blank rows
    rowlines = [rowlines[0]] + [row for row in rowlines[1:] if row != '' and not row.startswith('#')]

    reader = csv.DictReader(rowlines)
    rows = [row for row in reader]
    # DictReader has fieldnames attribute
    cols = reader.fieldnames

    return rows, cols

def parse_yaml(yamlfile):
    res = None
    with open(yamlfile, 'r') as yamlf:
        res = yaml.safe_load(yamlf)
    return res

def print_csv(outcols: KeysView[str] | list[str], outrows: list[dict[str, Any]], filename: Path | str):
    with open(filename, 'w', newline='') as ocsv:
        writer = csv.DictWriter(ocsv, fieldnames=outcols)
        writer.writeheader()
        for xrow in outrows:
            writer.writerow(xrow)

def print_json(jsdata, jsfilename):
    with open(jsfilename, 'w') as jsf:
        json.dump(jsdata, jsf)

def print_yaml(obj, ofilename):
    yaml_str = yaml.dump(obj)
    with open(ofilename, 'w', newline='') as YF:
        YF.write(yaml_str)

