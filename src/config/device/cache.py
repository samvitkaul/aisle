
from loguru import logger
from pydantic import AliasChoices, BaseModel, Field, PositiveInt

from ..knob import KnobVal

INFO    = logger.info
DEBUG   = logger.debug
ERROR   = logger.error
WARNING = logger.warning

class Cache(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    name                   : str
    num_banks              : PositiveInt = 1
    bytes_per_clk_per_bank : PositiveInt = Field(alias="bpc")
    size_per_bank          : KnobVal     = Field(validation_alias=AliasChoices("size", "sz"), serialization_alias="sz")

RegFile = Cache
