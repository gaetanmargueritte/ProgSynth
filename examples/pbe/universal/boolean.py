from typing import Callable, Dict
from examples.pbe.universal.dsl_factory import DSLFactory
from synth.syntax import auto_type

# simple boolean arithmetic

__syntax = {
    "and": "bool -> bool -> bool",
    "or": "bool -> bool -> bool",
    "xor": "bool -> bool -> bool",
    "not": "bool -> bool",
    "=>": "bool -> bool -> bool",
    "=": "bool -> bool -> bool",
    "distinct": "bool -> bool -> bool",
    "ite": "bool -> bool -> bool -> bool",
}

__semantic = {
    "and": lambda x: lambda y: x and y,
    "or": lambda x: lambda y: x or y,
    "xor": lambda x: lambda y: x != y,
    "not": lambda x: not(x),
    "=>": lambda x: lambda y: not x or y,
    "=": lambda x: lambda y: x == y,
    "distinct": lambda x: lambda y: x != y,
    "ite": lambda b: lambda x: lambda y: x if b else y,
}

class BooleanDSL(DSLFactory):
    def _add_options(self) -> Dict[str, Callable]:
        return {}
    
__lexicon = [0,1]
factory = BooleanDSL(__syntax, __semantic, __lexicon)