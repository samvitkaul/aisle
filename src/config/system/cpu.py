
from loguru import logger
from pydantic import BaseModel

INFO    = logger.info

class CPU(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    name       : str
