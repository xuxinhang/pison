# ## LALR Test Fixture ## #
from enum import Enum
from com import AUG_SYMBOL_EOF, AUG_SYMBOL_ERROR
from lalr import GrammarLalr


class SN(Enum):
    SS = 100
    S = 101
    L = 102
    R = 103


class ST(Enum):
    ID = 201
    EQ = 202
    STAR = 203


ex_productions = [
    (SN.SS, SN.S),
    (SN.S, SN.L, ST.EQ, SN.R),
    (SN.S, SN.R),
    (SN.L, ST.STAR, SN.R),
    (SN.L, ST.ID),
    (SN.R, SN.L),
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

grm = GrammarLalr()
grm.set_grammar(productions=grm_productions,
                terminals=grm_terminals,
                nonterminals=grm_nonterminals)

grm.items_lr0()
grm.print_lr0_itemset_collection()
grm.print_lr0_kernel_collection()

grm.discover_lookahead()
grm.print_lookahead_propagate_table()
grm.print_lookahead_generate_table()

grm.propagate_lookahead()
grm.print_lookahead_generate_table()

