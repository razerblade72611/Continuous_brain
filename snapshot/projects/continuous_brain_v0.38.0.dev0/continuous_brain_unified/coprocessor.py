from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any,Protocol
import json,time

@dataclass(frozen=True)
class CoprocessorRequest:
    purpose:str; payload:dict[str,Any]
@dataclass(frozen=True)
class CoprocessorResponse:
    available:bool; payload:dict[str,Any]=field(default_factory=dict); provider:str="disabled"
class Coprocessor(Protocol):
    def invoke(self,request:CoprocessorRequest)->CoprocessorResponse: ...
class NullCoprocessor:
    def invoke(self,request:CoprocessorRequest)->CoprocessorResponse: return CoprocessorResponse(False,{},"disabled")
class LoggingCoprocessor:
    def __init__(self,inner:Coprocessor,log_path:str|Path): self.inner=inner; self.log_path=Path(log_path)
    def invoke(self,request:CoprocessorRequest)->CoprocessorResponse:
        response=self.inner.invoke(request); self.log_path.parent.mkdir(parents=True,exist_ok=True)
        with open(self.log_path,'a',encoding='utf-8') as f:
            f.write(json.dumps({"time":time.time(),"purpose":request.purpose,"request":request.payload,"response_available":response.available,"provider":response.provider},sort_keys=True)+'\n')
        return response
