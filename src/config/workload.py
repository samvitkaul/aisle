
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional

WLType = Literal['BTEN', 'JSON', 'ONNX']

class WLInfo(BaseModel, extra='forbid', populate_by_name=True, frozen=True, arbitrary_types_allowed=True):
    wltype    : str
    wlname    : str
    basedir   : str
    source    : str
    wli_name  : str
    wli_params: dict
    wli_inputs: dict[str, Any] = Field(default_factory=dict)
    batchsize : int = 1


    def with_batchsize(self, bs: int) -> 'WLInfo':
        return self.model_copy(update({'batchsize': bs}))

    def with_inputs(self, ix: dict) -> 'WLInfo':
        return self.model_copy(update({'wli_inputs': dict(ix)}))

    def __str__(self):
        xparts = [
                f'{self.wltype.lower()}',
                f'{self.wlame}',
                f'{self.wli_name}',
                f'{self.batchsize}',
                ]
        xstr = '-'.join(xparts)
        return xstr
