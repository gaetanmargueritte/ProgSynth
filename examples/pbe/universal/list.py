from typing import Callable, Dict
from examples.pbe.universal.dsl_factory import DSLFactory

__syntax = {
    "head": "'a list -> 'a optional",
    "tail": "'a list -> 'a list",
    "length": "'a list -> int",
    "insert": "'a list -> 'a -> 'a list",
    "append": "'a list -> 'a -> 'a list",
    "extend": "'a list -> 'a list -> 'a list",
    "pop": "'a list -> 'a optional",
    "count": "'a list -> 'a -> int"
}


__semantics = {
    "head": lambda l: l.pop(0) if len(l) > 0 else None,
    "tail": lambda l: l[1:] if len(l) > 0 else [],
    "length": lambda l: len(l),
    "insert": lambda l: lambda x: l.insert(0, x),
    "append": lambda l: lambda x: l.append(x),
    "extend": lambda l: lambda l2: l.extend(l2),
    "pop": lambda l: l.pop() if len(l) > 0 else None,
    "count": lambda l: lambda x: l.count(x)
}

class ListDSL(DSLFactory):
    def _add_options(self) -> Dict[str, Callable]:
        return {}
    
__lexicon = []
factory = ListDSL(__syntax, __semantics, __lexicon)