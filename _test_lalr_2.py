# ## LALR Test Fixture ## #
from enum import Enum
from com import AUG_SYMBOL_EOF, AUG_SYMBOL_ERROR
from lalr import GrammarLalr


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

grm = GrammarLalr()
grm.set_grammar(productions=grm_productions,
                terminals=grm_terminals,
                nonterminals=grm_nonterminals)

grm.items_lr0()
grm.print_lr0_kernel_collection()

grm.discover_lookahead()
grm.print_lookahead_propagate_table()
grm.print_lalr_kernel_collection()

grm.propagate_lookahead()
grm.print_lalr_kernel_collection()

grm.lalr_items()
grm.print_lalr_itemset_collection()

grm.construct_parsing_table()
grm.print_parsing_table()

