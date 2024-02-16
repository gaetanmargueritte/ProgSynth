from typing import Dict, Callable, Optional, Tuple, List, Union, Any
import numpy as np
from synth import Task, PBE, Example
from importlib import import_module
from examples.pbe.universal.dsl_factory import DSLFactory
from synth.semantic.evaluator import DSLEvaluator, Evaluator
from synth.generation.sampler import LexiconSampler, UnionSampler
from synth.pbe.task_generator import (
    TaskGenerator,
    basic_output_validator,
    reproduce_dataset,
)
from synth.syntax import (
    DSL,
    auto_type,
    Program
)
from synth.task import Dataset
import sys
from progpysmt.smtlib.parser import ProgSmtLibParser
from progpysmt.pslobject import PSLObject
from progpysmt.exceptions import ProgPysmtConstraintError, PysmtSyntaxError
import pathlib

DATASET_PATH = "examples/pbe/universal/dataset"
MODULE_PATH =  'examples.pbe.universal.'
CSTE = 'cste'
SMT_TO_TYPE_CASTING: Dict[str, str] = {
    "int": lambda x: int(x),
    "float": lambda x: float(x),
    "bool": lambda x: boolchecker(str.lower(x)),
    "_bitvec64": lambda x: int("0x" + x[2:], 16),
    "_bitvec32": lambda x: int("0x" + x[2:], 16),
    "_bitvec16": lambda x: int("0x" + x[2:], 16),
    "_bitvec8": lambda x: int("0x" + x[2:], 16),
    "_bitvec4": lambda x: int("0x" + x[2:], 16),
}
SMT_TO_DSL: Dict[str, List[str]] = {
    "LIA": ("int", ["ite"]),
}
known_logics: List[str] = ['float', 'int', 'list', 'str', 'bitvector', 'boolean']
known_glues: Dict[str, str] = {
    "int": "float",
    "str": "int"
    }

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
        else:
            raise ValueError("Unknown logic %s" % s)
    for s, l in known:
        if s in known_glues and known_glues[s] in known_names:
            known.append(('glue' + s + '2' + known_glues[s], []))
    dsl, evaluator, lexicon, constants = __load_known_logics(known)
    # we do not manage yet unknown cases
    return dsl, evaluator, lexicon, constants



def boolchecker(input: str) -> bool:
    if input == "true":
        return True
    elif input == "false":
        return False
    raise ValueError("Bool object is neither True or False: %s"%input)

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
    evaluator = DSLEvaluator(semantics) 
    return dsl, evaluator, lexicon, constants

def search_lexicon(lexicon: List[Any]):
    if len(lexicon) == 0:
        print("A lexicon is required!", file=sys.stderr)
    if len(lexicon) == 1:
        return lexicon[0]
    minval = min(lexicon[0])
    maxval = max(lexicon[0])
    steps = len(lexicon[0])
    precision = 1
    for l in lexicon[1:]:
        lmin = min(l)
        lmax = max(l)
        lsteps = len(l)
        if minval > lmin:
            minval = lmin
        if maxval < lmax:
            maxval = lmax
        if steps < lsteps:
            steps = lsteps
    if steps > (maxval-minval):
        precision = precision/(10*(round(steps/(maxval-minval))))
    return [np.arange(minval, maxval, precision)]

def interpret(text: Optional[str], dsl: DSL, pslobject: PSLObject) -> Program:
    """interprets a string representing a program given a dsl and a pslobject"""
    def cast(value: str, type: str):
        if type == "int":
            return int(value)
        if type == "float":
            return float(value)
        return value
    if text is None:
        return None
    import re
    type_request = auto_type(pslobject.type)
    text = re.sub(r'\( ', '(', text)
    text = re.sub(r' \)', ')', text)
    cst = {}
    for constant in pslobject.constants:
        for val in pslobject.constants[constant]:
            cst[val] = (auto_type(constant), cast(val, constant))
    prog = dsl.parse_program(text, type_request, cst)
    return prog

def specify(f: str, inputs: List[str], output: List[str], type: str) -> Tuple[Any, Any]:
    type_list = type.split(' -> ')
    typed_inputs = []
    for x,y in zip(inputs, type_list):
        typed_inputs.append(SMT_TO_TYPE_CASTING[str.lower(y)](x))
    typed_output = SMT_TO_TYPE_CASTING[str.lower(type_list[-1])](output)
    return (typed_inputs, typed_output)

def deobfuscate(f: str, object: PSLObject) -> Tuple[Any, Any]:
    for c in object.constraints:
        args = c.table_of_symbols
        obfuscated = False
        for func in c.pbe:
            for a in args:
                if a in func and args[a] == None:
                    return '', True
        #body_spl = c.pbe.split(' ')
    return '', False

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

def get_sygus_dataset():
    path = pathlib.Path("dataset/")
    dataset_files = path.rglob("*sl")
    all_logics = [("boolean", [])]
    tasks = []
    parser = ProgSmtLibParser()
    objects: List[PSLObject] = []
    failures = 0
    failure_syntax = 0
    failure_obf = 0
    maybe_obf = 0
    total = 0
    for f in dataset_files:
        total += 1
        file = open(f, 'r')
        print(f)
        try:
            pslobject: PSLObject = parser.get_script(file, f)
        except ProgPysmtConstraintError as e:
            print("Constraint error on %s\t%s"%(f,e), file=sys.stderr)
            failures += 1
        except PysmtSyntaxError as e:
            print("Syntax error on %s\t%s"%(f,e), file=sys.stderr)
            failure_syntax += 1
        for logic in pslobject.logics:
            if logic == ("BV", []):
                type_split = pslobject.type.split(' -> ')
                for t in type_split:
                    bv_split = t.split('_BitVec')
                    if len(bv_split) > 1:
                        x = bv_split[1]
                        new_logic = ('bitvector' , [x, "ite"])
                        if new_logic not in all_logics:
                            all_logics.append(new_logic)
            elif logic not in all_logics:
                all_logics.append(logic)
        if pslobject.smtlogic != '' and pslobject.smtlogic not in all_logics:
            s, l = SMT_TO_DSL[pslobject.smtlogic]
            if (s, l) not in all_logics:
                all_logics.append((s, l))
        objects.append(pslobject)
    dsl, evaluator, lexicon, constants = load_logics(all_logics)
    lexicon = search_lexicon(lexicon)
    obfuscated = []
    for o in objects:
        examples = []
        if len(o.pbe) > 0:
            typed_pbe = [specify(o.filename, x, y, o.type) for x, y in o.pbe]
            examples = [Example(input, output) for input, output in typed_pbe]
        else:
            if any([1 if len(c.pbe) <= 0 else 0 for c in o.constraints]):
                print("\tCould not find any possible PBE using synthetized method %s in file %s" % (o.func_name, o.filename), file=sys.stderr)
                failure_obf += 1
            else:
                ret, obf = deobfuscate(o.filename, o)
                obfuscated.append(o)
                if obf:
                    failure_obf += 1
                else:
                    maybe_obf += 1
        """
        print(o.func_name) 
        print("*"*50)
        print(o.solution)
        print("*"*50)
        print(o.constants)
        print("*"*50)
        print(o.type)
        print("*"*75)
        print(o.pbe)
        print("*"*75)
        
        for item in o.grammar_interpretation:
            print(item)
            for bidule in o.grammar_interpretation[item]:
                print("\t-" + str(bidule))
        print("\n\n")
        """
        if len(examples) > 0:
            # TODO?
            #if len(o.constants.keys()) > 0:
            #    task = Task[PBEWithConstants]

            for met, type in o.methods:
                v1, v2 = o.methods[(met, type)]
                print(v1)
                print(v2)
                print(met + " " + type + " " + v1)
            task = Task[PBE](auto_type(o.type), PBE(examples), interpret(o.solution, dsl, o), {"name": o.func_name, 'cfg': (o.grammar_interpretation, o.func_param), "methods": o.methods, "file": o.filename})
            tasks.append(task)

    for d in obfuscated[55:60]:
        print(f"{d.filename}, {d.func_name}")
    print(evaluator)
    dataset = Dataset(tasks, metadata={"dataset": "sygus", "logics": all_logics})
    print("SUCCESS %d |||| FAILURES %d |||| SYNTAX FAILURES %d |||| OBFUSCATION FAILURES %d |||| SOLVABLE OBFUSCATIONS %d |||| TOTAL %d" % (total-failures-failure_syntax-failure_obf-maybe_obf, failures, failure_syntax, failure_obf, maybe_obf, total))
    return dataset


if __name__ == "__main__":
    dataset = get_sygus_dataset()
    dataset.save("sygus.pickle")
    #for d in dataset[55:60]:
    #    print(d)
