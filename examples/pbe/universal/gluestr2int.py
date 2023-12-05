from typing import Callable, Dict
from examples.pbe.universal.dsl_factory import DSLFactory

__syntax = {
    "from_int": "int -> string",
    "from_code": "int -> string",
    "to_int": "string -> int",
    "to_code": "string -> int",
}


__semantics = {
    "from_int": lambda x: str(x) if x > 0 else "",
    "from_code": lambda x: chr(x) if 0x00000 <= x and x <= 0x2FFFF else "",
    "to_int": lambda x: int(x),
    "to_code": lambda x: ord(x) if len(x) == 1 else ""
}

class GLUEStr2IntDSL(DSLFactory):
    def _add_options(self) -> Dict[str, Callable]:
        return {}
    
__lexicon = []
factory = GLUEStr2IntDSL(__syntax, __semantics, __lexicon)
