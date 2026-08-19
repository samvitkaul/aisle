
from pydantic import BaseModel
from loguru   import logger

INFO    = logger.info

class CPU(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    name       : str
