from typing import Dict, Callable, Optional, Tuple, List, Union, Any
import numpy as np
from synth import Task, PBE
from importlib import import_module
from examples.pbe.universal.dsl_factory import DSLFactory
from synth.semantic.evaluator import DSLEvaluator, DSLEvaluatorWithConstant, Evaluator
from synth.generation.sampler import LexiconSampler, UnionSampler
from synth.pbe.task_generator import (
    TaskGenerator,
    basic_output_validator,
    reproduce_dataset,
)
from synth.syntax import (
    DSL,
    Arrow,
    FixedPolymorphicType,
    PrimitiveType,
    FunctionType,
    auto_type,
)
from synth.task import Dataset

MODULE_PATH =  'examples.pbe.universal.'
CSTE = 'cste'
known_logics: List[str] = ['float', 'int', 'list', 'str']
known_glues: Dict[str, str] = {
    "int": "float",
    "str": "int"
    }

def __load_known_logics(logics: List[Tuple[str, List[str]]]):
    cste_flag = False
    modules = [import_module(MODULE_PATH + s) for s, _ in logics]
    factory_list: List[DSLFactory] = [mod.factory for mod in modules]

    f_init = factory_list[0]
    _, options = logics[0]
    f_init.call_options(options)
    constants = []
    if CSTE in options:
        cste_flag = True
        if len(f_init.get_lexicon()) > 0: 
            constants.extend(f_init.get_constant_types())
    dsl = f_init.get_dsl()
    semantics = f_init.get_semantics()
    lexicon = [f_init.get_lexicon()]
    keys_ptr = 1
    for f in factory_list[1:]:
        _, options = logics[keys_ptr]
        keys_ptr += 1
        f.call_options(options)
        if CSTE in options:
            cste_flag = True
            if len(f.get_constant_types()) > 0:
                constants.extend(f.get_constant_types())

        dsl = dsl | f.get_dsl()
        semantics.update(f.get_semantics())
        if len(f.get_lexicon()) > 0: 
            lexicon.append(f.get_lexicon())
    dsl.instantiate_polymorphic_types(len(logics) + 1)
    if cste_flag == True:
        evaluator = DSLEvaluatorWithConstant(semantics, set(constants))
    else:
        evaluator = DSLEvaluator(semantics)
    return dsl, evaluator, lexicon


    

#def __load_known_logics1(logics: List[str]) -> Tuple[DSL, DSLEvaluator] :
#    modules = [import_module(s) for s in logics]
#    semantics_list = [mod.semantics for mod in modules]
#    dsl_list: List[DSL] = [mod.dsl for mod in modules]
#    dsl = dsl_list[0]
#    semantics: Dict = semantics_list[0]
#    for d, s in zip(dsl_list[1:], semantics_list):
#        dsl = dsl | d
#        semantics.update(s)
#
#    dsl.instantiate_polymorphic_types(len(logics))
#    evaluator = DSLEvaluator(semantics)
#    return dsl, evaluator
#
# checks if called logics are known, in which case we directly import, otherwise maybe use dsl_loader methods
# Just a dummy for a more refined version of a loader
def load_logics(logics: List[Tuple[str, List[str]]]):
    known = []
    known_names = []
    unknown = []
    for s, l in logics:
        if s in known_logics and s not in known:
            known.append((s, l))
            known_names.append(s)
        elif s not in unknown:
            unknown.append((s, l))
    for s, l in known:
        if s in known_glues and known_glues[s] in known_names:
            known.append(('glue' + s + '2' + known_glues[s], []))
    dsl, evaluator, lexicon = __load_known_logics(known)
    # we do not manage yet unknown cases
    return dsl, evaluator, lexicon

logics = [
    ('float', ['ite']), 
    ('int', ['cste']),
    ('list', [])
]
dsl, evaluator, lexicon = load_logics(logics)
lexicon = lexicon[0]

def reproduce_universal_dataset(
    dataset: Dataset[PBE],
    dsl: DSL,
    evaluator: Evaluator,
    seed: Optional[int] = None,
    int_bound: int = 1000,
    *args: Any,
    **kwargs: Any
) -> Tuple[TaskGenerator, List[int]]:

    int_range: List[int] = [int_bound, 0]
    int_range[1] = -int_range[0]

    float_range: List[float] = [float(int_bound), 0]
    float_range[1] = -float_range[0]
    float_bound = float(int_bound)

    def analyser(start: None, element: Union[int, float]) -> None:
        if isinstance(element, int):
            int_range[0] = min(int_range[0], max(-int_bound, element))
            int_range[1] = max(int_range[1], min(int_bound, element))
        elif isinstance(element, float):
            float_range[0] = min(float_range[0], max(-float_bound, element))
            float_range[1] = max(float_range[1], min(float_bound, element))

    def get_element_sampler(start: None) -> UnionSampler:
        int_lexicon = list(range(int_range[0], int_range[1] + 1))
        float_lexicon = [
            round(x, 1) for x in np.arange(float_range[0], float_range[1] + 1, 0.1)
        ]
        return UnionSampler(
            {
                auto_type('int'): LexiconSampler(int_lexicon, seed=seed),
                auto_type('bool'): LexiconSampler([True, False], seed=seed),
                auto_type('float'): LexiconSampler(float_lexicon, seed=seed),
            }
        )

    def get_validator(start: None, max_list_length: int) -> Callable[[Any], bool]:
        return basic_output_validator(
            {
                int: list(range(int_range[0], int_range[1] + 1)),
                float: [
                    round(x, 1)
                    for x in np.arange(float_range[0], float_range[1] + 1, 0.1)
                ],
            },
            max_list_length,
        )

    def get_lexicon(start: None) -> List[float]:
        return [round(x, 1) for x in np.arange(float_range[0], float_range[1] + 1, 0.1)]

    return reproduce_dataset(
        dataset,
        dsl,
        evaluator,
        None,
        analyser,
        get_element_sampler,
        get_validator,
        get_lexicon,
        seed,
        *args,
        **kwargs
    )



#dsl.parse_program("()")
#print(dsl)