# ## Test Fixture ## #
from enum import Enum
from com import AUG_SYMBOL_EOF, AUG_SYMBOL_ERROR
from clr import GrammarClr


class SN(Enum):
    SS = 100
    S = 101
    C = 102


class ST(Enum):
    c = 201
    d = 202


ex_productions = [
    (SN.SS, SN.S),
    (SN.S, SN.C, SN.C),
    (SN.C, ST.c, SN.C),
    (SN.C, ST.d)
]


def digitalize_production(prod, terminals, nonterminals):
    def dg(s):
        try:
            return nonterminals.index(s)
        except Exception:
            return ~terminals.index(s)

    return tuple(map(dg, prod))


grm_terminals = [AUG_SYMBOL_EOF, AUG_SYMBOL_ERROR] + list(ST)
grm_nonterminals = list(SN)
grm_productions = list(map(
    lambda p: digitalize_production(p, terminals=grm_terminals, nonterminals=grm_nonterminals),
    ex_productions
))

grm = GrammarClr()
grm.set_grammar(productions=grm_productions,
                terminals=grm_terminals,
                nonterminals=grm_nonterminals)
grm.items()
grm.print_itemset_collection()

grm.construct_table()
grm.print_analysis_table()

