from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class SeedRange:
    start:int; end:int; purpose:str; status:str="RESERVED"; source:str=""
    def overlaps(self,other:'SeedRange')->bool: return not (self.end < other.start or other.end < self.start)

class SeedRegistry:
    def __init__(self,path:str|Path): self.path=Path(path)
    def load(self)->list[SeedRange]:
        if not self.path.exists(): return []
        d=json.loads(self.path.read_text()); return [SeedRange(**x) for x in d.get('ranges',[])]
    def assert_available(self,start:int,end:int)->None:
        candidate=SeedRange(start,end,'candidate')
        hits=[r for r in self.load() if r.overlaps(candidate)]
        if hits: raise ValueError('seed range overlaps: '+', '.join(f'{r.start}-{r.end}:{r.purpose}' for r in hits))
    def reserve(self,r:SeedRange)->None:
        if r.start>r.end: raise ValueError('start > end')
        self.assert_available(r.start,r.end); ranges=self.load()+[r]
        self.path.write_text(json.dumps({'ranges':[x.__dict__ for x in ranges]},indent=2,sort_keys=True)+'\n')
