from typing import Any, Callable, Dict, List
from examples.pbe.universal.dsl_factory import DSLFactory
from synth.syntax import DSL, auto_type
from operator import attrgetter

__syntax = {
    "bvadd": lambda x: f"{x} -> {x} -> {x}",
    "bvmul": lambda x: f"{x} -> {x} -> {x}",
    "bvudiv": lambda x: f"{x} -> {x} -> {x}",
    "bvurem": lambda x: f"{x} -> {x} -> {x}",
    #"-": lambda x: f"{x} -> {x} -> {x}",
    "bvneg": lambda x: f"{x} -> {x} -> {x}",
    "bvnot": lambda x: f"{x} -> {x} -> {x}",
    "bvor": lambda x: f"{x} -> {x} -> {x}",
    "bvand": lambda x: f"{x} -> {x} -> {x}",
    #"xor": lambda x: f"{x} -> {x} -> {x}",
    "bvshl": lambda x: f"{x} -> {x} -> {x}",
    "bvlshr": lambda x: f"{x} -> {x} -> {x}"
}

# strong bit detection required with mbv
# modulo mbv in order to make sure we never leave max bitvector
# TODO: two's complement
__semantics = {
    "bvadd": lambda mbv: lambda x: lambda y: x+y if x+y < mbv and x+y > -mbv else -mbv+((x+y+1)%mbv) if x+y >= mbv else ((x+y-2)%mbv),
    "bvmul": lambda mbv: lambda x: lambda y: x*y if x*y < mbv and x*y > -mbv else -mbv+((x*y+1)%mbv) if x*y >= mbv else (((x*y)-2)%mbv),
    "bvudiv": lambda mbv: lambda x: lambda y: 1 if y == 0 else x//y if x//y < mbv and x//y > -mbv else -mbv+(((x//y)+1)%mbv) if x*y >= mbv else (((x//y)-2)%mbv),
    "bvurem": lambda mbv: lambda x: lambda y: x if y == 0 else x%y,
    #"-": lambda mbv: lambda x: lambda y: x-y if x-y < mbv and x-y > -mbv else -mbv+((x-y+1)%mbv) if x-y >= mbv else ((x-y-2)%mbv),
    "bvneg": lambda mbv: lambda x: mbv - x,
    "bvnot": lambda mbv: lambda x: -x if x < mbv and x > -mbv else -(x%mbv),
    "bvor": lambda mbv: lambda x: lambda y: x|y,
    "bvand": lambda mbv: lambda x: lambda y: x&y, 
    #"xor": lambda mbv: lambda x: lambda y: x^y,
    "bvshl": lambda mbv: lambda x: lambda n: x<<n if x << n < mbv else mbv-((x<<n)%mbv), 
    "bvlshr": lambda mbv: lambda x: lambda n: x >> n if x >> n> -mbv else 0,
}

class BVChild(DSLFactory):
    __ite_syntax = {
        "ite": lambda x: f"bool -> {x} -> {x} -> {x}",
        "bvult": lambda x: f"{x} -> {x} -> bool",
        "=": lambda x: f"{x} -> {x} -> bool"
        #">": lambda x: f"{x} -> {x} -> bool",
        #"<=": lambda x: f"{x} -> {x} -> bool",
        #">=": lambda x: f"{x} -> {x} -> bool",
        #"=": lambda x: f"{x} -> {x} -> bool",
    }
    __ite_semantics = {
        "ite": lambda b: lambda x: lambda y: x if b else y,
        "bvult": lambda x: lambda y: x < y,
        "=": lambda x: lambda y: x == y,
        #">": lambda x: lambda y: x > y,
        #"<=": lambda x: lambda y: x <= y,
        #">=": lambda x: lambda y: x >= y,
        #"=": lambda x: lambda y: x == y,
    }

    def __init__(self, syntax: Dict[str, Any], semantics: Dict[str, Any], lexicon: List[Any], x: int, forbidden_pattern: Dict[str, str] = None, needs_init: bool = False) -> None:
        super().__init__(syntax, semantics, lexicon, forbidden_pattern)
        self._type = "bv" + str(x)
        self._num_bits = 1 << x - 1
        for rule in self._syntax:
            self._syntax[rule] = self._syntax[rule](self._type)
            self._semantics[rule] = self._semantics[rule](self._num_bits)

    def _add_ite(self):
        new_ite = {}
        for rule in self.__ite_syntax:
            new_ite[rule] = self.__ite_syntax[rule](self._type)
        self._syntax.update(new_ite)
        self._semantics.update(self.__ite_semantics)

    def _add_cste(self):
        self._constant_types(auto_type(self._type))

    def _add_options(self) -> Dict[str, Callable]:
        return {'ite': self._add_ite,
                'cste': self._add_cste}


class BVFactory(DSLFactory):

    __instances: List[BVChild] = []

    def __init__(self, 
                 syntax: Dict[str, str], 
                 semantics: Dict[str, Any], 
                 lexicon: List[Any], 
                 forbidden_pattern: Dict[str, str] = None, 
                 needs_init: bool = True) -> None:
        super().__init__(syntax, semantics, lexicon, forbidden_pattern, needs_init)
        self._blank_syntax = syntax.copy()
        self._blank_semantics = semantics.copy()
        self._dsl = None
        self.__instances = []

    def get_dsl(self) -> DSL:
        return self._dsl

    def __glue_all(self):
        def reduce_bv(bv: int, max_bit: int) -> None:
            prev_pow = max_bit // 2
            if bv%max_bit >= prev_pow:
                return max_bit-(bv%max_bit)
            return bv%max_bit
        
        self.__instances.sort(key=attrgetter('_num_bits'))
        first = self.__instances[0]
        for next in self.__instances[1:]:
            keysmall2large = "" + first._type + "2" + next._type
            keylarge2small = "" + next._type + "2" + first._type
            glue_syntax = {
                keysmall2large: f"{first._type} -> {next._type}",
                keylarge2small: f"{next._type} -> {first._type}"
            
            }
            glue_semantics = {
                keysmall2large: lambda x: x if x >= 0 else next.__num_bits + x,
                keylarge2small: lambda x: reduce_bv(x)
            }
            self._syntax.update(glue_syntax)
            self._semantics.update(glue_semantics)
            first = next


    def _initx(self, x: int):
        self.__instances.append(BVChild(self._blank_syntax.copy(), self._blank_semantics.copy(), self._lexicon, x, self._forbidden_pattern, False))

    def _add_ite(self):
        for i in self.__instances:
            i._add_ite()

    def _add_cste(self):
        for i in self.__instances:
            i._add_cste()

    def _add_init_options(self) -> Dict[str, Callable]:
        return {'4': lambda: self._initx(4),
                '8': lambda: self._initx(8),
                '16': lambda: self._initx(16),
                '32': lambda: self._initx(32),
                '64': lambda: self._initx(64),
                'default': lambda: self._initx(8)}

    def _add_options(self) -> Dict[str, Callable]:
        return {'ite': self._add_ite,
                'cste': self._add_cste}
    
    def _finalize(self) -> None:
        self._syntax.clear()
        self._semantics.clear()
        instantiated_semantics = {}
        first = self.__instances[0]
        first._finalize()
        self._dsl = first.get_dsl()
        instantiated_semantics.update(first._semantics)
        for i in self.__instances[1:]:
            i._finalize()
            idsl = i.get_dsl()
            instantiated_semantics.update(i._semantics)
            self._dsl = self._dsl | idsl
        if len(self.__instances) > 1:
            self._semantics.clear()
            self._syntax.clear()
            self.__glue_all()
            glue_dsl = DSL(auto_type(self._syntax), self._forbidden_pattern)
            glue_sem = glue_dsl.instantiate_semantics(self._semantics)
            self._dsl = self._dsl | DSL(auto_type(self._syntax), self._forbidden_pattern)
            self._dsl.instantiate_polymorphic_types(2)
            instantiated_semantics.update(glue_sem)
        self._semantics = instantiated_semantics
        
__lexicon = [x for x in range(-255,256)]
factory = BVFactory(__syntax, __semantics, __lexicon)
