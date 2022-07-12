from abc import ABC, abstractmethod
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Generic,
    Union,
)

from synth.syntax.grammars.grammar import Grammar
from synth.syntax.program import Constant, Function, Primitive, Program, Variable
from synth.syntax.type_system import Arrow, Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
T = TypeVar("T")

DerivableProgram = Union[Primitive, Variable, Constant]


class DetGrammar(Grammar, ABC, Generic[U, V, W]):
    def __init__(
        self,
        start: Tuple[Type, U],
        rules: Dict[Tuple[Type, U], Dict[DerivableProgram, V]],
        clean: bool = True,
    ):
        self.start = start
        self.rules = rules
        if clean:
            self.clean()
        self.type_request = self._guess_type_request_()

    def __hash__(self) -> int:
        return hash((self.start, str(self.rules)))

    def __rule_to_str__(self, P: DerivableProgram, out: V) -> str:
        return "{}: {}".format(P, out)

    def __str__(self) -> str:
        s = f"Print a {self.name()}\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                out = self.rules[S][P]
                s += "   {}\n".format(self.__rule_to_str__(P, out))
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def _guess_type_request_(self) -> Type:
        """
        Guess the type request of this grammar.
        """
        # Compute the type request
        type_req = self.start[0]
        variables: List[Variable] = []
        for S in self.rules:
            for P in self.rules[S]:
                if isinstance(P, Variable):
                    if P not in variables:
                        variables.append(P)
        n = len(variables)
        for i in range(n):
            j = n - i - 1
            for v in variables:
                if v.variable == j:
                    type_req = Arrow(v.type, type_req)
        return type_req

    def __contains__(self, program: Program) -> bool:
        return self.__contains_rec__(program, self.start, self.start_information())[0]

    def __contains_rec__(
        self, program: Program, start: Tuple[Type, U], information: W
    ) -> Tuple[bool, W, Tuple[Type, U]]:
        if start not in self.rules:
            return False, information, start
        if isinstance(program, Function):
            function = program.function
            args_P = program.arguments
            if function not in self.rules[start]:
                return False, information, start
            information, next = self.derive(information, start, function)  # type: ignore
            for arg in args_P:
                contained, information, next = self.__contains_rec__(
                    arg, start=next, information=information
                )
                if not contained:
                    return False, information, next
            return True, information, next
        elif isinstance(program, (Primitive, Variable, Constant)):
            if program not in self.rules[start]:
                return False, information, start
            information, next = self.derive(information, start, program)
            return True, information, next
        return False, information, start

    def clean(self) -> None:
        """
        Clean this deterministic grammar by removing non reachable, non producible.
        """
        self._remove_non_productive_()
        self._remove_non_reachable_()

    @abstractmethod
    def _remove_non_reachable_(self) -> None:
        pass

    @abstractmethod
    def _remove_non_productive_(self) -> None:
        pass

    @abstractmethod
    def derive(
        self, information: W, S: Tuple[Type, U], P: DerivableProgram
    ) -> Tuple[W, Tuple[Type, U]]:
        """
        Given the current information and the derivation S -> P, produces the new information state and the next S after this derivation.
        """
        pass

    def derive_all(
        self,
        information: W,
        S: Tuple[Type, U],
        P: Program,
        current: Optional[List[Tuple[Type, U]]] = None,
    ) -> Tuple[W, List[Tuple[Type, U]]]:
        """
        Given current information and context S, produces the new information and all the contexts the grammar went through to derive program P.
        """
        current = current or []
        if isinstance(P, (Primitive, Variable, Constant)):
            information, ctx = self.derive(information, S, P)
            current.append(ctx)
            return (information, current)

        elif isinstance(P, Function):
            F = P.function
            information, _ = self.derive_all(information, S, F, current)
            S = current[-1]
            for arg in P.arguments:
                information, _ = self.derive_all(information, S, arg, current)
                S = current[-1]
            return (information, current)
        assert False

    @abstractmethod
    def arguments_length_for(self, S: Tuple[Type, U], P: DerivableProgram) -> int:
        pass

    @abstractmethod
    def start_information(self) -> W:
        pass

    def reduce_derivations(
        self,
        reduce: Callable[[T, Tuple[Type, U], DerivableProgram, V], T],
        init: T,
        program: Program,
        start: Optional[Tuple[Type, U]] = None,
    ) -> T:
        """
        Reduce the given program using the given reduce operator.

        reduce is called after derivation.
        """

        return self.__reduce_derivations_rec__(
            reduce, init, program, start or self.start, self.start_information()
        )[0]

    def __reduce_derivations_rec__(
        self,
        reduce: Callable[[T, Tuple[Type, U], DerivableProgram, V], T],
        init: T,
        program: Program,
        start: Tuple[Type, U],
        information: W,
    ) -> Tuple[T, W, Tuple[Type, U]]:
        value = init
        if isinstance(program, Function):
            function = program.function
            args_P = program.arguments
            information, next = self.derive(information, start, function)  # type: ignore
            value = reduce(value, start, function, self.rules[start][function])  # type: ignore
            for arg in args_P:
                value, information, next = self.__reduce_derivations_rec__(
                    reduce, value, arg, start=next, information=information
                )
            return value, information, next
        elif isinstance(program, (Primitive, Variable, Constant)):
            information, next = self.derive(information, start, program)
            value = reduce(value, start, program, self.rules[start][program])
            return value, information, next
        return value, information, start
