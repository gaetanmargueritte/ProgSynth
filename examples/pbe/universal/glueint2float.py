from typing import Callable, Dict
from examples.pbe.universal.dsl_factory import DSLFactory
import numpy as np

__syntax = {
    "int2float": "int -> float",
    "float2int": "float -> int"
}


__semantics = {
    "int2float": lambda x: float(x),
    "float2int": lambda x: int(x)
}

class GLUEInt2FloatDSL(DSLFactory):
    def _add_options(self) -> Dict[str, Callable]:
        return {}
    
__lexicon = []
factory = GLUEInt2FloatDSL(__syntax, __semantics, __lexicon)