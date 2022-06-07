from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

import vose

from synth.syntax.concrete.concrete_cfg import ConcreteCFG, NonTerminal
from synth.syntax.program import Constant, Function, Primitive, Program, Variable
from synth.syntax.type_system import Arrow

PRules = Dict[NonTerminal, Dict[Program, Tuple[List[NonTerminal], float]]]


class ConcretePCFG:
    """
    Object that represents a probabilistic context-free grammar
    with normalised weights

    rules: a dictionary of type {S: D}
    with S a non-terminal and D a dictionary : {P : l, w}
    with P a program, l a list of non-terminals, and w a weight
    representing the derivation S -> P(S1, S2, ...) with weight w for l' = [S1, S2, ...]

    list_derivations: a dictionary of type {S: l}
    with S a non-terminal and l the list of programs P appearing in derivations from S,
    sorted from most probable to least probable

    max_probability: a dictionary of type {S: (Pmax, probability)} cup {(S, P): (Pmax, probability)}
    with S a non-terminal

    hash_table_programs: a dictionary {hash: P}
    mapping hashes to programs
    for all programs appearing in max_probability
    """

    def __init__(
        self,
        start: NonTerminal,
        rules: PRules,
        max_program_depth: int,
        clean: bool = False,
    ):
        self.start = start
        self.rules = rules
        self.max_program_depth = max_program_depth
        self.hash_table_programs: Dict[int, Program] = {}

        if clean:
            self.clean()

        # Compute the type request
        type_req = self.start.type
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
        self.type_request = type_req
        self.ready_for_sampling: bool = False

    def clean(self) -> None:
        """
        Remove non productive.
        Remove non reachable
        Normalise probabilities
        Sort rules
        """
        self.__remove_non_productive__()
        self.__remove_non_reachable__()
        self.__normalise__()
        self.__sort__()

    def __hash__(self) -> int:
        return hash((self.start, str(self.rules), self.max_program_depth))

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, ConcretePCFG)
            and self.type_request == o.type_request
            and self.rules == o.rules
        )

    def __str__(self) -> str:
        s = "Print a PCFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                s += "   {} - {}: {}     {}\n".format(P, P.type, args_P, w)
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def init_sampling(self, seed: Optional[int] = None) -> None:
        self.ready_for_sampling = True
        self.vose_samplers = {}
        self.list_derivations = {}

        for i, S in enumerate(self.rules):
            self.list_derivations[S] = sorted(
                self.rules[S], key=lambda P: self.rules[S][P][1]
            )
            self.vose_samplers[S] = vose.Sampler(
                np.array(
                    [self.rules[S][P][1] for P in self.list_derivations[S]], dtype=float
                ),
                seed=seed + i if seed else None,
            )

    def __sort__(self) -> None:
        for S in self.rules:
            sorted_derivation_list = sorted(
                self.rules[S], key=lambda P: -self.rules[S][P][1]
            )
            new_rules = {}
            for P in sorted_derivation_list:
                new_rules[P] = self.rules[S][P]
            self.rules[S] = new_rules

    def __normalise__(self) -> None:
        for S in self.rules:
            s = sum([self.rules[S][P][1] for P in self.rules[S]])
            for P in list(self.rules[S].keys()):
                args_P, w = self.rules[S][P]
                self.rules[S][P] = (args_P, w / s)

    def return_unique(self, P: Program) -> Program:
        """
        ensures that if a program appears in several rules,
        it is represented by the same object
        """
        h = hash(P)
        if h in self.hash_table_programs:
            return self.hash_table_programs[h]
        else:
            self.hash_table_programs[h] = P
            return P

    def __remove_non_productive__(self) -> None:
        """
        remove non-terminals which do not produce programs
        """
        new_rules: PRules = {}
        for S in reversed(self.rules):
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                if all([arg in new_rules for arg in args_P]) and w > 0:
                    if S not in new_rules:
                        new_rules[S] = {}
                    new_rules[S][P] = self.rules[S][P]

        for S in set(self.rules):
            if S in new_rules:
                self.rules[S] = new_rules[S]
            else:
                del self.rules[S]

    def __remove_non_reachable__(self) -> None:
        """
        remove non-terminals which are not reachable from the initial non-terminal
        """
        reachable: Set[NonTerminal] = set()
        reachable.add(self.start)

        reach: Set[NonTerminal] = set()
        new_reach: Set[NonTerminal] = set()
        reach.add(self.start)

        for _ in range(self.max_program_depth):
            new_reach.clear()
            for S in reach:
                for P in self.rules[S]:
                    args_P, _ = self.rules[S][P]
                    for arg in args_P:
                        new_reach.add(arg)
                        reachable.add(arg)
            reach.clear()
            reach = new_reach.copy()

        for S in set(self.rules):
            if S not in reachable:
                del self.rules[S]

    def sampling(self) -> Generator[Program, None, None]:
        """
        A generator that samples programs according to the PCFG G
        """
        assert self.ready_for_sampling
        while True:
            yield self.sample_program(self.start)

    def sample_program(self, S: Optional[NonTerminal] = None) -> Program:
        assert self.ready_for_sampling
        S = S or self.start
        i: int = self.vose_samplers[S].sample()
        P = self.list_derivations[S][i]
        args_P, _ = self.rules[S][P]
        if len(args_P) == 0:
            return P
        arguments = []
        for arg in args_P:
            arguments.append(self.sample_program(arg))
        return Function(P, arguments)

    def probability(self, P: Program, S: Optional[NonTerminal] = None) -> float:
        """
        Compute the probability of a program P generated from the non-terminal S
        """
        S = S or self.start
        if isinstance(P, Function):
            F = P.function
            args_P = P.arguments
            probability = self.rules[S][F][1]

            for i, arg in enumerate(args_P):
                probability *= self.probability(arg, self.rules[S][F][0][i])
            return probability

        elif isinstance(P, (Variable, Primitive)):
            return self.rules[S][P][1]

        return 0

    def __contains__(self, P: Program) -> bool:
        return self.probability(P) > 0

    @classmethod
    def from_weights(
        cls,
        cfg: ConcreteCFG,
        get_weight: Callable[
            [NonTerminal, Union[Primitive, Variable, Constant]], float
        ],
    ) -> "ConcretePCFG":
        augmented_rules: PRules = {}
        for S in cfg.rules:
            augmented_rules[S] = {}
            for P in cfg.rules[S]:
                augmented_rules[S][P] = (cfg.rules[S][P], get_weight(S, P))
        return ConcretePCFG(
            start=cfg.start,
            rules=augmented_rules,
            max_program_depth=cfg.max_program_depth,
            clean=True,
        )

    @classmethod
    def from_samples(
        cls,
        cfg: ConcreteCFG,
        samples: Iterable[Program],
    ) -> "ConcretePCFG":
        rules_cnt: Dict[NonTerminal, Dict[Program, int]] = {}
        for S in cfg.rules:
            rules_cnt[S] = {}
            for P in cfg.rules[S]:
                rules_cnt[S][P] = 0

        def add_count(S: NonTerminal, P: Program) -> bool:
            if isinstance(P, Function):
                F = P.function
                args_P = P.arguments
                success = add_count(S, F)

                for i, arg in enumerate(args_P):
                    add_count(cfg.rules[S][F][i] if success else S, arg)  # type: ignore
            else:
                if P not in rules_cnt[S]:
                    # This case occurs when a forbidden pattern has been removed from the CFG
                    # What to do? Ignore for now, but this bias a bit the probabilities
                    # TODO: perhaps rethink that? or provide a program simplifier
                    return False
                else:
                    rules_cnt[S][P] += 1
            return True

        for sample in samples:
            add_count(cfg.start, sample)

        # Remove null derivations to avoid divide by zero exceptions when normalizing later
        for S in cfg.rules:
            total = sum(rules_cnt[S][P] for P in cfg.rules[S])
            if total == 0:
                del rules_cnt[S]

        return ConcretePCFG(
            start=cfg.start,
            rules={
                S: {P: (cfg.rules[S][P], rules_cnt[S][P]) for P in rules_cnt[S]}  # type: ignore
                for S in rules_cnt
            },
            max_program_depth=cfg.max_program_depth,
            clean=True,
        )

    @classmethod
    def uniform(cls, cfg: ConcreteCFG) -> "ConcretePCFG":
        return cls.from_weights(cfg, lambda _, __: 1)
