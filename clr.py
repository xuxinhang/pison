from com import AUG_SYMBOL_EOF


class GrammarBase():
    pass


class GrammarClr(GrammarBase):
    def __init__(self):
        # grammar definition
        self.prods = []
        self.terminals = []
        self.nonterminals = []
        # grammar table
        self._table_goto = None
        self._table_action = None

    def set_grammar(self, *, productions, terminals, nonterminals):
        self.prods = productions
        self.terminals = terminals
        self.nonterminals = nonterminals
        self.compute_grammar_cache()

    def compute_grammar_cache(self):
        _prod_map = self._prod_map = [[] for _ in self.nonterminals]
        for prod_idx, prod_exp in enumerate(self.prods):
            _prod_map[prod_exp[0]].append(prod_idx)

        self.terminal_map = {'EOF': ~self.terminals.index(AUG_SYMBOL_EOF)}
        self._first_map = {}

    def _first(self, X, lookup_cache=True, trace=None):
        if lookup_cache and X in self._first_map:
            return self._first_map[X]
        if trace is None:
            trace = []

        # X is a terminal symbol
        if X < 0:
            fst = self._first_map[X] = [X]
            return fst

        # X is an nonterminal symbol
        fst = []
        for p in (p for p in self.prods if p[0] == X):
            # for empty productions
            if len(p) == 1 or p[1] is None:
                fst.append(None)
                continue
            # Remember to skip left-recursive productions
            fst += self._first_beta(p[1:], left=X, trace=trace+[X])

        self._first_map[X] = fst
        return fst

    def _first_beta(self, beta, left=None, trace=None):
        ret = []
        for X in beta:
            if X == left or (trace is not None and X in trace):
                break
            X_first = self._first(X, trace=trace)
            ret += (s for s in X_first if s is not None)
            if None not in X_first:
                break
        else:
            ret.append(None)
        return ret

    def closure(self, I):
        I = I[:]
        v, w = 0, len(I)
        while v < w:
            prod, dot_pos, lasym = I[v]
            prod_exp = self.prods[prod]
            if dot_pos < len(prod_exp):
                firsts = self._first_beta(prod_exp[dot_pos+1:] + (lasym,))
                dot_sym = prod_exp[dot_pos]
                if dot_sym >= 0:
                    dot_prods = self._prod_map[dot_sym]
                    for y in dot_prods:
                        for b in firsts:
                            if (new_item := (y, 1, b)) not in I:
                                I.append(new_item)
                                w += 1
            v += 1
        I.sort(key=lambda t: (t[0], t[1], ~t[2]))
        return I

    def goto(self, I, X):
        J = []
        for prod, dot_pos, lasym in I:
            prod_exp = self.prods[prod]
            if prod_exp[dot_pos] == X:
                J.append((prod, dot_pos+1, lasym))
        return self.closure(J)

    def items(self):
        C = [self.closure([(0, 1, self.terminal_map['EOF'])])]

        v, w = 0, len(C)
        while v < w:
            I = C[v]
            # seek possible goto symbol
            dot_symbols = []
            for prod, dot_pos, lasym in I:
                prod_exp = self.prods[prod]
                if dot_pos < len(prod_exp):
                    if (s := prod_exp[dot_pos]) not in dot_symbols:
                        dot_symbols.append(s)
            # compute and append new item sets
            for X in dot_symbols:
                g = self.goto(I, X)
                if len(g) and g not in C:
                    C.append(g)
                    w += 1
            v += 1

        self.itemset_collection = C  # "collection"?
        return C

    def stringify_item(self, item):
        prod, dot_pos, lasym = item
        prod_exp = self.prods[prod]

        def wrap_symbol(s):
            if s < 0:
                return '[' + str(self.terminals[~s]) + ']'
            else:
                return '(' + str(self.nonterminals[s]) + ')'

        ss = []
        ss.append('%s -> ' % wrap_symbol(prod_exp[0]))
        for i, r in enumerate(prod_exp[1:]):
            ss.append(('\u25AA' if i + 1 == dot_pos else ' ') + str(wrap_symbol(r)))
        ss.append('\u25AA' if dot_pos == len(prod_exp) else ' ')

        return ''.join(ss) + ' , ' + wrap_symbol(lasym)

    def print_itemset_collection(self):
        for i, itemset in enumerate(self.itemset_collection):
            print(f'C[{i}]')
            print(*(' '*4 + self.stringify_item(t) for t in itemset), sep='\n')


### Test Fixture ###
from enum import Enum
from com import AUG_SYMBOL_EOF, AUG_SYMBOL_ERROR


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


def digitalize_production(prod, terminals=[], nonterminals=[]):
    def dg(s):
        try:
            return nonterminals.index(s)
        except Exception:
            return ~terminals.index(s)

    return tuple(map(dg, prod))


grm_terminals =  [AUG_SYMBOL_EOF, AUG_SYMBOL_ERROR] + list(ST)
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

