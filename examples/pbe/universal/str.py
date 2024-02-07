from synth.syntax import DSL, auto_type
from typing import Callable, Dict, Optional
import re
from examples.pbe.universal.dsl_factory import DSLFactory

def _indexof(s1: str, s2: str, index: Optional[int]) -> int:
    try:
        return s1.index(s2, index)
    except:
        return -1
    
def _replace(s: str, t: str, t2: str) -> str:
    if len(t) == 0:
        return t2 + s
    else:
        return s.replace(t, t2)


__syntax = {
    "++": "string -> string -> string",
    "at": "string -> int -> string",
    "substr": "string -> int -> int -> string",
    "indexof": "string -> string -> int optional-> int",
    "replace": "string -> string -> string -> string",
    "none": "RegLan",
    "all": "RegLan",
    "allchar": "RegLan",
    "++": "RegLan -> RegLan -> RegLan",
    "union": "RegLan -> RegLan -> RegLan",
    "inter": "RegLan -> RegLan -> RegLan",
    "*": "RegLan -> RegLan",
    "+": "RegLan -> RegLan",
    "opt": "RegLan -> RegLan",
    "len": "string -> int",
}

__semantics = {
    "++": lambda x: lambda y: "" + x + y,
    "at": lambda x: lambda pos: x[pos] if pos > 0 and pos < len(x) else "",
    "substr": lambda x: lambda left: lambda right: x[left:right] if left < right and left > 0 and right < len(x) else "",
    "indexof": lambda x: lambda y: lambda pos: _indexof(x, y, pos),
    "replace": lambda x: lambda y: lambda z: _replace(x, y, z),
    "none": r"^\s*$",
    "all": r"[\s\S]*",
    "allchar": r"[\s\S]{1}",
    "++": lambda x: lambda y: x + y,
    "union": lambda x: lambda y: r'|'.join([x, y]),
    "inter": lambda x: lambda y: r''.join([r'(?=', x, r')(?=', y, r')']),
    "*": lambda x: x + r'*',
    "+": lambda x: x + r'+',
    "opt": lambda x: r'|'.join([x,r'']),
    "len": lambda x: len(x)
}

class StrDSL(DSLFactory):
    __ite_syntax = {
        "ite": "bool -> 'a[string|int] -> 'a[string|int] -> 'a[string|int]",
        "in_re": "string -> RegLan -> bool",
        "contains": "string -> string -> bool",
        "prefixof": "string -> string -> bool",
        "suffixof": "string -> string -> bool",
        "<": "string -> string -> bool",
        "<=": "string -> string -> bool",
        "is_digit": "string -> bool",
    }


    __ite_semantics = {
        "ite": lambda b: lambda x: lambda y: x if b else y,
        "in_re": lambda str: lambda pattern: True if re.search(pattern, str) else False,
        "contains": lambda x: lambda y: y in x,
        "prefixof": lambda x: lambda y: x.startswith(y),
        "suffixof": lambda x: lambda y: x.endswith(y),
        "<": lambda x: lambda y: x < y,
        "<=": lambda x: lambda y: x <= y,
        "is_digit": lambda x: x.isdigit()
    }


    def _add_ite(self):
        self._syntax.update(self.__ite_syntax)
        self._semantics.update(self.__ite_semantics)


    def _add_cste(self):
        self._constant_types = {auto_type("string")}

    def _add_options(self) -> Dict[str, Callable]:
        return {"ite": self._add_ite,
                }
    
__lexicon = list([chr(i) for i in range(32, 126)])
factory = StrDSL(__syntax, __semantics, __lexicon)
