from typing import Dict, Any, Callable, List, Set
from synth.syntax import DSL, Type, auto_type
from abc import ABC, abstractmethod
import sys

class DSLFactory(ABC):
    @abstractmethod
    def _add_options(self) -> Dict[str, Callable]:
        pass

    def _add_init_options(self) -> Dict[str, Callable]:
        return {}

    def __init__(self, 
                 syntax: Dict[str, str], 
                 semantics: Dict[str, Any], 
                 lexicon: List[Any],
                 forbidden_pattern: Dict[str, str] = None,
                 needs_init: bool = False) -> None:
        self._options: Dict[str, Callable] = self._add_options()
        self._init_options: Dict[str, Callable] = self._add_init_options()
        self._syntax = syntax
        self._semantics = semantics
        self._lexicon = lexicon
        self._forbidden_pattern = forbidden_pattern
        self._constant_types = {}
        self._needs_init = needs_init
        self._dsl: DSL = None

    def known_options(self) -> List[str]:
        return self._options
    
    def get_dsl(self) -> DSL:
        return self._dsl
    
    def get_semantics(self) -> Dict[str, Any]:
        return self._semantics
    
    def get_constant_types(self) -> List[str]:
        return self._constant_types
    
    def get_lexicon(self) -> List[Any]:
        return self._lexicon

    def _finalize(self) -> None:
        self._dsl = DSL(auto_type(self._syntax), self._forbidden_pattern)
        self._dsl.instantiate_polymorphic_types(1)
        self._semantics = self._dsl.instantiate_semantics(self._semantics)
    
    # double pass through list may be improved in future updates
    def call_options(self, options: List[str]) -> None:
        try:
            if self._needs_init and len(self._init_options) > 0:
                for o in options:
                    if o in self._init_options:
                        self._init_options[o]()
            elif self._needs_init:
                self._init_options["default"]()
            for o in options:
                if o in self._options:
                    self._options[o]()
        except Exception as e:
            print("Unexpected option from list: %s, %s"% (o,e), file=sys.stderr)
            print(e, file=sys.stderr)
            raise # prints in stderr

        self._finalize()