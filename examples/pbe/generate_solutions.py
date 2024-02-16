import os
import sys
import csv
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, Set, Dict

import tqdm

from progpysmt.pslobject import Method
from examples.pbe.universal.universal import load_logics
from synth import Dataset, PBE
from synth.semantic.evaluator import DSLEvaluator
from examples.pbe.dataset_loader import DatasetUnpickler
from synth import Task, PBE
import examples.pbe.universal as universal

from synth.pbe.solvers import (
    NaivePBESolver,
    PBESolver,
    CutoffPBESolver,
    ObsEqPBESolver,
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
    for solver in [NaivePBESolver, CutoffPBESolver, ObsEqPBESolver]
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
    help="ngram used by grammars (default: 2)",
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

#def save(trace: Iterable) -> None:
#    with open(file, "w") as fd:
#        writer = csv.writer(fd)
#        writer.writerows(trace)

def produce_cfg_from_task(
    task: Task[PBE],
    original_dsl: DSL,
    evaluator: DSLEvaluator,
) -> CFG:
    def __to_semantic(method_body: str, method_param: Dict[str, str], method_signature: str):
        met_spl = method_body.split(' ')
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
                continue
            prim = dsl.get_primitive(tok)
            if prim is not None:
                prim_stack.append(prim)
                args_stack.append(prim.type.arguments())
                res += tok + " "
            elif tok in method_param:
                # tok is a var
                if tok not in vars2ptr:
                    varptr += 1
                    vars2ptr[tok] = "var" + str(varptr-1)
                res += vars2ptr[tok] + " "
            else:
                # tok is a constant
                current_prim_type = args_stack.pop()
                argtype = current_prim_type.pop(0)
                cast = SMT_TO_TYPE_CASTING[str.lower(argtype.type_name)](tok)
                cons_dict[str(cast)] = (cast, argtype.type_name)
                res += cast + " "
        return res[:-1], cons_dict
    def __lambdify(program: Program, vars: List[int]):
        def __lambda(met: Callable):
            return lambda x: x
        if len(vars) == 0:
            return program
        varnum = vars.pop()
        varname = "var" + str(varnum)
        print(program.pretty_print())
        evaluation = evaluator.eval()
        for subprog in program.depth_first_iter():
            
            print(subprog, type(subprog))
        # if arity > len(accu)
            # lambda x: accu.append(x)
        # if arity <= len(accu)
            # return evaluator.eval(program, *accu)
        assert 0


    dsl = original_dsl
    cfg, func_param = task.metadata["cfg"]
    methods = task.metadata["methods"]
    metnames = [metname for metname, _ in task.metadata["methods"].keys()]
    print(task.metadata["file"])
    if len(cfg) > 0:
        new_dsl_syntax = {}
        new_dsl_semantic = {}
        for c in cfg:
            varname, vartype = c
            print(varname, " ", vartype)
            for rule in cfg[c]:
                print(rule)
                if type(rule) == str:
                    if rule in func_param:
                        continue
                    else:
                        new_dsl_syntax[rule] = vartype
                        new_dsl_semantic[rule] = SMT_TO_TYPE_CASTING[str.lower(vartype)](rule) 
                elif type(rule) == Method:
                    if dsl.get_primitive(rule.name):
                        continue
                    elif rule.name in metnames:
                        met_body, met_param = methods[(rule.name, rule.signature)]
                        tokens, constants = __to_semantic(met_body, met_param, rule.signature)
                        program = dsl.parse_program(tokens, auto_type(rule.signature), constants)
                        vars = sorted(list(program.used_variables()))
                        program = __lambdify(program, vars)
                        #print(program.used_variables())
                        assert 0
                    
                    print("\t" + rule.name + "\t" + rule.signature)
        return

    

def produce_pcfgs(
        full_dataset: Dataset[PBE],
        dsl: DSL,
        evaluator: DSLEvaluator,
) -> Union[List[ProbDetGrammar], List[ProbUGrammar]]:
    all_type_requests = (
        full_dataset.type_requests()
    )

    for t in full_dataset.tasks:
        produce_cfg_from_task(t, dsl, evaluator)
    cfgs = [
        produce_cfg_from_task(t, dsl)
        for t in full_dataset.tasks
    ]

def enumerative_search(
        dataset: Dataset[PBE],
        evaluator: DSLEvaluator,
        pcfgs: Union[List[ProbDetGrammar], List[ProbUGrammar]],
        trace: List[Tuple[bool, float]],
        solver: PBESolver,
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
    stats_name = solver.available_stats()
    if start == 0:
        trace.append(["solved", "solution"] + stats_name)
    for task, pcfg in zip(tasks[start:], pcfgs[start:]):
        if task.metadata.get("name", None) is not None:
            pbar.set_description(task.metadata["name"])
        total += 1
        task_solved = False
        solution = None
        try:
            sol_generator = solver.solve(
                task, custom_enumerate(pcfg), timeout=task_timeout
            )
            solution = next(sol_generator)
            task_solved = True
            solved += 1
        except KeyboardInterrupt:
            break
        except StopIteration:
            pass
        out = [task_solved, solution] + [solver.get_stats(name) for name in stats_name]
        solver.reset_stats()
        trace.append(out)
        pbar.update(1)
        evaluator.clear_cache()
        if i%10 == 0:
            pbar.set_postfix_str("Saving...")
            #save(trace)
        pbar.set_postfix_str(f"Solver {solved}/{total}")
    pbar.close()

if __name__ == "__main__":
    dataset, dsl, evaluator = load_dsl_and_dataset()
    produce_pcfgs(dataset, dsl, evaluator)