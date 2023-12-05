from typing import Callable, Dict
from examples.pbe.universal.dsl_factory import DSLFactory
from synth.syntax import auto_type

## bitvector: faire avec des int infinis pythons

__syntax = {
    "+": "int -> int -> int",
    "-": "int -> int -> int",
    "*": "int -> int -> int",
    "div": "int -> int -> int",
    "mod": "int -> int -> int",
    "abs": "int -> int"
}

__semantics = {
    "+": lambda x: lambda y: x+y,
    "-": lambda x: lambda y: x-y,
    "*": lambda x: lambda y: x*y,
    "div": lambda x: lambda y: 0 if y == 0 else int(x/y),
    "mod": lambda x: lambda y: 0 if y == 0 else x%y,
    "abs": lambda x: abs(x),
}

class IntDSL(DSLFactory):
    __ite_syntax = {
        "ite": "bool -> int -> int -> int",
        "=": "int -> int -> bool",
        "<=": "int -> int -> bool",
        ">=": "int -> int -> bool",
        "<": "int -> int -> bool",
        ">": "int -> int -> bool",
    }

    __ite_semantics = {
        "ite": lambda b: lambda x: lambda y: x if b else y,
        "=": lambda x: lambda y: x==y,
        "<=": lambda x: lambda y: x<=y,
        ">=": lambda x: lambda y: x>=y,
        "<": lambda x: lambda y: x<y,
        ">": lambda x: lambda y: x>y
    }


    def _add_ite(self):
        self._syntax.update(self.__ite_syntax)
        self._semantics.update(self.__ite_semantics)

    
    def _add_cste(self):
        self._constant_types = {auto_type("int")}

    def _add_options(self) -> Dict[str, Callable]:
        return {"ite": self._add_ite,
                "cste": self._add_cste}

__lexicon = [x for x in range(-256, 256 + 1)]
factory = IntDSL(__syntax, __semantics, __lexicon)