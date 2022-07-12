from synth.syntax.grammars.cfg import CFG
from synth.syntax.dsl import DSL
from synth.syntax.program import Primitive
from synth.syntax.type_system import (
    INT,
    STRING,
    Arrow,
    FunctionType,
    List,
    PolymorphicType,
    PrimitiveType,
)


syntax = {
    "+": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}


def test_from_dsl() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = CFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        for rule in cfg.rules:
            assert rule[1][0][1] <= max_depth
            for P in cfg.rules[rule]:
                if isinstance(P, Primitive):
                    assert P.primitive != "non_reachable"
                    assert P.primitive != "non_productive"
                    assert P.primitive != "head"
                else:
                    assert P.type == INT


def test_function_as_variable() -> None:
    dsl = DSL(syntax)
    max_depth = 5
    cfg = CFG.from_dsl(dsl, FunctionType(Arrow(INT, INT), INT), max_depth)
    assert cfg.size() > 0


def test_clean() -> None:
    dsl = DSL(syntax)
    for max_depth in [3, 7, 11]:
        cfg = CFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        for rule in cfg.rules:
            assert rule[1][0][1] <= max_depth
            for P in cfg.rules[rule]:
                if isinstance(P, Primitive):
                    assert P.primitive != "non_reachable"
                    assert P.primitive != "non_productive"

        cpy = CFG.from_dsl(dsl, FunctionType(INT, INT), max_depth)
        cpy.clean()
        assert cfg == cpy
