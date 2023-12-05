from typing import Callable, Dict
from examples.pbe.universal.dsl_factory import DSLFactory
from synth.syntax import auto_type
import numpy as np

__syntax = {
    ".+": "float -> float -> float",
    ".-": "float optional -> float -> float",
    ".*": "float -> float -> float",
    ".div": "float -> float -> float",
    ".mod": "float -> float -> float",
    ".abs": "float -> float",
}

__semantics = {
    ".+": lambda x: lambda y: round(x + y, 1),
    ".-": lambda x: lambda y: round(x - y, 1) if x is not None else -y,
    ".*": lambda x: lambda y: round(x * y, 1),
    ".div": lambda x: lambda y: 0 if y == 0 else x/y,
    ".mod": lambda x: lambda y: x if y == 0 else x%y,
    ".abs": lambda x: abs(x),
}

class FloatDSL(DSLFactory):
    __ite_syntax = {
        ".ite": "bool -> float -> float -> float",
        ".=": "float -> float -> bool",
        ".<=": "float -> float -> bool",
        ".>=": "float -> float -> bool",
        ".<": "float -> float -> bool",
        ".>": "float -> float -> bool",
    } 

    __ite_semantics = {
        ".ite": lambda b: lambda x: lambda y: x if b else y,
        ".=": lambda x: lambda y: x==y,
        ".<=": lambda x: lambda y: x<=y,
        ".>=": lambda x: lambda y: x>=y,
        ".<": lambda x: lambda y: x<y,
        ".>": lambda x: lambda y: x>y
    }


    def _add_ite(self):
        self._syntax.update(self.__ite_syntax)
        self._semantics.update(self.__ite_semantics)

    def _add_cste(self):
        self._constant_types = {auto_type("float")}

    def _add_options(self) -> Dict[str, Callable]:
        return {"ite": self._add_ite,
                "cste": self._add_cste}
    
__lexicon = [round(x, 1) for x in np.arange(-256, 256 + 1, 0.1)]
factory = FloatDSL(__syntax, __semantics, __lexicon)
