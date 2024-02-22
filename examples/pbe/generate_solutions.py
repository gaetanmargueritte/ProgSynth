import os
import sys
import csv
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, Set, Dict

import tqdm

from progpysmt.pslobject import Method
from examples.pbe.universal.universal import load_logics, get_types_from_sygus
from synth import Dataset, PBE
from synth.semantic.evaluator import DSLEvaluator
from examples.pbe.dataset_loader import DatasetUnpickler
from synth import Task, PBE
import examples.pbe.universal as universal

from synth.pbe.solvers import (
    NaivePBESolver,
    PBESolver,
    CutoffPBESolver,
    RestartPBESolver,
)
from synth.syntax import (
    ProbDetGrammar,
    ProbUGrammar,
    Program,
    Variable,
    DSL,
    Type,
    CFG, 
    UCFG,
    auto_type,
    hs_enumerate_prob_grammar,
    bs_enumerate_prob_grammar,
    bps_enumerate_prob_grammar,
    hs_enumerate_prob_u_grammar,
    hs_enumerate_bucket_prob_grammar,
    hs_enumerate_bucket_prob_u_grammar,
    ProgramEnumerator,
)

import argparse

def boolchecker(input: str) -> bool:
    if input == "true":
        return True
    elif input == "false":
        return False
    raise ValueError("Bool object is neither True or False: %s"%input)

SMT_TO_TYPE_CASTING: Dict[str, str] = {
    "int": lambda x: int(x),
    "float": lambda x: float(x),
    "bool": lambda x: boolchecker(str.lower(x)),
    "_bitvec64": lambda x: int("0x" + x[2:], 16),
    "_bitvec32": lambda x: int("0x" + x[2:], 16),
    "_bitvec16": lambda x: int("0x" + x[2:], 16),
    "_bitvec8": lambda x: int("0x" + x[2:], 16),
    "_bitvec4": lambda x: int("0x" + x[2:], 16),
    "bv64": lambda x: int("0x" + x[2:], 16),
    "bv32": lambda x: int("0x" + x[2:], 16),
    "bv16": lambda x: int("0x" + x[2:], 16),
    "bv8": lambda x: int("0x" + x[2:], 16),
    "bv4": lambda x: int("0x" + x[2:], 16),
}
SOLVERS = {
    solver.name(): solver
    for solver in [NaivePBESolver, CutoffPBESolver]
}
base_solvers = {x: y for x, y in SOLVERS.items()}

SEARCH_ALGOS = {
    "beap_search": (bps_enumerate_prob_grammar, None),
    "heap_search": (hs_enumerate_prob_grammar, hs_enumerate_prob_u_grammar),
    "bucket_search": (
        lambda x: hs_enumerate_bucket_prob_grammar(x, 3),
        lambda x: hs_enumerate_bucket_prob_u_grammar(x, 3),
    ),
    "bee_search": (bs_enumerate_prob_grammar, None),
}

parser = argparse.ArgumentParser(
    description="Solve program from cfg", fromfile_prefix_chars="@"
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="{dsl_name}.pickle",
    help="the dataset file to load (default: {dsl_name}.pickle)",
)
parser.add_argument(
    "-s",
    "--search",
    choices=SEARCH_ALGOS.keys(),
    default=list(SEARCH_ALGOS.keys())[0],
    help=f"enumeration algorithm (default: {list(SEARCH_ALGOS.keys())[0]})",
)
parser.add_argument(
    "--solver",
    choices=list(SOLVERS.keys()),
    default="naive",
    help=f"used solver (default: naive)",
)
parser.add_argument(
    "-o", "--output", type=str, default="./", help="output folder (default: './')"
)
parser.add_argument(
    "-t", "--timeout", type=float, default=300, help="task timeout in s (default: 300)"
)
parser.add_argument(
    "--max-depth",
    type=int,
    default=5,
    help="maximum depth of grammars used (-1 for infinite, default: 5)",
)
parser.add_argument(
    "--ngram",
    type=int,
    default=2,
    choices=[1, 2],
    help="ngram used by grammars (default: 1)",
)
parameters = parser.parse_args()
dataset_file: str = parameters.dataset
search_algo: str = parameters.search
method: Callable[[Any], PBESolver] = SOLVERS[parameters.solver]
output_folder: str = parameters.output
task_timeout: float = parameters.timeout
max_depth: int = parameters.max_depth
ngram: int = parameters.ngram


if not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
    print("Dataset must be a valid dataset file!", file=sys.stderr)
    sys.exit(1)

custom_enumerate, _ = SEARCH_ALGOS[search_algo]

start_index = (
    0
    if not os.path.sep in dataset_file
    else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
)
dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]

supported_type_requests = None

def load_dsl_and_dataset() -> Tuple[Dataset[PBE], DSL, DSLEvaluator]:
    dataset: Dataset = Dataset.load(dataset_file, DatasetUnpickler)
    logics = dataset.metadata["logics"]
    dsl, evaluator, lexicon, constants = load_logics(logics)
    return dataset, dsl, evaluator

def save(trace: Iterable) -> None:
    with open(file, "w") as fd:
        writer = csv.writer(fd)
        writer.writerows(trace)

def get_primitive_with_type(name: str, dsl: DSL, type: List[Type]):
    for P in dsl.list_primitives:
        type_set, type_set_polymorphic = P.type.decompose_type()
        if P.primitive == name and (all([m in type_set or m in type_set_polymorphic for m in type])):
            return P
    return None

def produce_cfg_from_task(
    task: Task[PBE],
    original_dsl: DSL,
    evaluator: DSLEvaluator,
) -> Tuple[CFG, DSLEvaluator]:
    def __to_semantic(method_body: str, method_param: Dict[str, str], method_signature: str):
        met_spl = method_body.split(' ')
        types_list = get_types_from_sygus(method_signature)
        types_list = [auto_type(t) for t in types_list]
        cons_dict = {}
        res = ""
        prim_stack = []
        args_stack = []
        vars2ptr = {}
        varptr = 0
        for tok in met_spl:
            if tok == '(':
                res += tok
                continue
            elif tok == ')':
                res = res[:-1] + ')' + ' '
                args_stack.pop()
                continue
            prim = get_primitive_with_type(tok, dsl, types_list)
            if prim is not None:
                # primitive exists, check for duplicate names 
                prim_stack.append(prim)
                args_stack.append(prim.type.arguments())
                res += tok + " "
            elif tok in method_param:
                # tok is a var
                current_prim_type = args_stack.pop()
                argtype = current_prim_type.pop(0)
                if tok not in vars2ptr:
                    varptr += 1
                    vars2ptr[tok] = "var" + str(varptr-1)
                res += vars2ptr[tok] + " "
                args_stack.append(current_prim_type)
            else:
                # tok is a constant
                current_prim_type = args_stack.pop()
                argtype = current_prim_type.pop(0)
                cast = SMT_TO_TYPE_CASTING[str.lower(argtype.type_name)](tok)
                cons_dict[str(cast)] = (auto_type(argtype.type_name), cast)
                res += str(cast) + " "
                args_stack.append(current_prim_type)
        return res[:-1], cons_dict
    def __lambdify(program: Program, vars: List[int]):
        def f(x: Any, n, acc):
            acc.append(x)
            if n == 1:
                return evaluator.eval(program, acc)
            return lambda x: f(x, n - 1, acc)
        return lambda x: f(x, len(vars), [])
    dsl = original_dsl
    cfg, func_param = task.metadata["cfg"]
    methods = task.metadata["methods"]
    metnames = [metname for metname, _ in task.metadata["methods"].keys()]
    constant_types = []
    print(task.metadata["file"])
    if len(cfg) > 0:
        new_dsl_syntax = {}
        new_dsl_semantic = {}
        for c in cfg:
            varname, vartype = c
            vartype = get_types_from_sygus(vartype)[0]
            for rule in cfg[c]:
                if type(rule) == str:
                    if rule in func_param:
                        continue
                    else:
                        cast = SMT_TO_TYPE_CASTING[vartype](rule)
                        new_dsl_syntax[str(cast)] = vartype
                        new_dsl_semantic[str(cast)] = cast 
                        constant_types.append(auto_type(vartype))
                elif type(rule) == Method:
                    str_types_list = get_types_from_sygus(rule.signature)
                    types_list = [auto_type(t) for t in str_types_list]
                    prim = get_primitive_with_type(rule.name, dsl, types_list)
                    if prim:
                        new_dsl_syntax[prim.primitive] = ' -> '.join(str_types_list)
                        new_dsl_semantic[prim.primitive] = evaluator.semantics[prim]
                    elif rule.name in metnames:
                        met_body, met_param = methods[(rule.name, rule.signature)]
                        tokens, constants = __to_semantic(met_body, met_param, rule.signature)
                        program = dsl.parse_program(tokens, auto_type(rule.signature), constants)
                        vars = sorted(list(program.used_variables()))
                        program = __lambdify(program, vars)
                        new_dsl_syntax[rule.name] = ' -> '.join(str_types_list)
                        new_dsl_semantic[rule.name] = program
                    #print("\t" + rule.name + "\t" + rule.signature)
    else:
        return (CFG.infinite(original_dsl, task.type_request, 2), evaluator)
    new_dsl = DSL(auto_type(new_dsl_syntax))
    new_dsl.instantiate_polymorphic_types(1)
    new_dsl_semantic = new_dsl.instantiate_semantics(new_dsl_semantic)
    return (CFG.infinite(new_dsl, task.type_request, 2), DSLEvaluator(new_dsl_semantic))

    

def produce_pcfgs(
        full_dataset: Dataset[PBE],
        dsl: DSL,
        evaluator: DSLEvaluator,
) -> Union[List[ProbDetGrammar], List[ProbUGrammar]]:
    cfgs = []
    evaluators = []
    for t in full_dataset.tasks:
        cfg, e = produce_cfg_from_task(t, dsl, evaluator)
        cfgs.append(cfg)
        evaluators.append(e)
    pcfgs = [ProbDetGrammar.uniform(c) for c in cfgs]
    return pcfgs, evaluators

def enumerative_search(
        dataset: Dataset[PBE],
        evaluators: List[DSLEvaluator],
        pcfgs: Union[List[ProbDetGrammar], List[ProbUGrammar]],
        trace: List[Tuple[bool, float]],
        custom_enumerate: Callable[
            [Union[ProbDetGrammar, ProbUGrammar]], ProgramEnumerator
        ]
) -> None:
    start = max(0, len(trace) - 1)
    pbar = tqdm.tqdm(total=len(pcfgs) - start, desc="Tasks", smoothing=0)
    i = 0
    solved = 0
    total = 0
    tasks = dataset.tasks
    init_solver = method(evaluators[0])
    if start == 0:
        trace.append(["filename", "solved", "solution"] + init_solver.available_stats())
    for task, pcfg, evaluator in zip(tasks[start:], pcfgs[start:], evaluators[start:]):
        #print("Solving %s"%task.metadata.get("file"))
        task_filename = task.metadata.get("file", None)
        solver: PBESolver = method(evaluator=evaluator)
        stats_name = solver.available_stats()
        if task_filename:
            pbar.set_description(str(task_filename))
        total += 1
        task_solved = False
        solution = None
        try:
            sol_generator = solver.solve(
                task, custom_enumerate(pcfg), timeout=task_timeout
            )
            solution = next(sol_generator)
            #print(solution)
            task_solved = True
            solved += 1
        except KeyboardInterrupt:
            break
        except StopIteration:
            pass
        out = [str(task_filename), task_solved, solution] + [solver.get_stats(name) for name in stats_name]
        solver.reset_stats()
        trace.append(out)
        pbar.update(1)
        evaluator.clear_cache()
        if i%10 == 0:
            pbar.set_postfix_str("Saving...")
            save(trace)
        pbar.set_postfix_str(f"Solver {solved}/{total}")
    pbar.close()

if __name__ == "__main__":
    dataset, dsl, evaluator = load_dsl_and_dataset()
    pcfgs, evaluators = produce_pcfgs(dataset, dsl, evaluator)
    done = 0
    tasks = dataset.tasks
    file = os.path.join(
        output_folder,
        f"{dataset_name}_{search_algo}_uniform.csv",
    )
    trace = []
    enumerative_search(dataset, evaluators, pcfgs, trace, custom_enumerate)
    save(trace)
    print("csv file was saved as:", file)

## TODO: Améliorer la trace
    # assurer les sémantiques de bv
    # lancer plafrim (30mn timeout)
    # faire la doc du parser avec des examples des dicos
    # ajouter le sampling
    # relancer plafrim ad vitam nauseam