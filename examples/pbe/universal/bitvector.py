from typing import Callable, Dict
from examples.pbe.universal.dsl_factory import DSLFactory
from synth.syntax import auto_type

__syntax = {
    "+": lambda x: f"{x} -> {x} -> {x}",
    "-": lambda x: f"{x} -> {x} -> {x}",
    "~": lambda x: f"{x} -> {x} -> {x}",
    "or": lambda x: f"{x} -> {x} -> {x}",
    "and": lambda x: f"{x} -> {x} -> {x}",
    "xor": lambda x: f"{x} -> {x} -> {x}",
    "<<": lambda x: f"{x} -> int -> {x}",
    ">>": lambda x: f"{x} -> int -> {x}"
}


# strong bit detection required with mbv
__semantics = {
    "+": lambda mbv: lambda x: lambda y: x+y if x+y < mbv else x+y - mbv,
    "-": lambda mbv: lambda x: lambda y: x-y if x-y > 0 #else 0,
    "~": lambda mbv: lambda x: -x, # utilize mbv
    "or": lambda mbv: lambda x: lambda y: x|y,
    "and": lambda mbv: lambda x: lambda y: x&y, 
    "xor": lambda mbv: lambda x: lambda y: x^y,
    "<<": lambda mbv: lambda x: lambda n: x*(2**n) if x*(2**n) < mbv #else mbv, 
    ">>": lambda mbv: lambda x: lambda n: x/(2**n) if x/(2**n) > 0 #else 0,    
}

class BVFactory(DSLFactory):
    __ite_syntax = {
     #todo if xynthia does it   
    }
    __ite_semantics = {
        # idem
    }
    __type = ""

    def _initx(self, x: int):
        self._type = "bv" + str(x)
        num_bits = 2**x
        for rule in self._syntax:
            self._syntax[rule] = self._syntax[rule](self._type)
            self._semantics[rule] = self._semantics[rule](num_bits)
        # should not be required, todo
        for rule in self.__ite_syntax:
            self.__ite_syntax[rule] = self.__ite_syntax[rule](self._type)
            self.__ite_semantics[rule] = self.__ite_semantics[rule](num_bits)
        
    def _add_ite(self):
        self._syntax.update(self.__ite_syntax)
        self._semantics.update(self.__ite_semantics)

    def _add_cste(self):
        self._constant_types = {auto_type(self.__type)}

    def _add_init_options(self) -> Dict[str, Callable]:
        return {'8': lambda: self._initx(8),
                '16': lambda: self._initx(16),
                '32': lambda: self._initx(32),
                '64': lambda: self._initx(64)}

    def _add_options(self) -> Dict[str, Callable]:
        return {'ite': self._add_ite,
                'cste': self._add_cste}