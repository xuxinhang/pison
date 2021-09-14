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
            if dot_pos < len(prod_exp) and prod_exp[dot_pos] == X:
                J.append((prod, dot_pos+1, lasym))
        return self.closure(J)

    def items(self):
        C = [self.closure([(0, 1, self.terminal_map['EOF'])])]

        v, w = 0, len(C)
        while v < w:
            I = C[v]
            # seek possible goto symbol
            dot_symbols = []
            for prod, dot_pos, _ in I:
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

    def construct_table(self):
        itemset_collection = self.itemset_collection

        n_s = len(self.itemset_collection)
        n_t = len(self.terminals)
        n_n = len(self.nonterminals)
        table_action = self._table_action =\
            memoryview(bytearray(n_s * n_t * 4)).cast('L', (n_s, n_t))
        table_goto = self._table_goto =\
            memoryview(bytearray(n_s * n_n * 4)).cast('L', (n_s, n_n))

        for i, I_i in enumerate(itemset_collection):
            for prod, dot_pos, lasym in I_i:
                prod_exp = self.prods[prod]
                # case: shift
                if dot_pos < len(prod_exp) and (a := prod_exp[dot_pos]) < 0:
                    I_j = self.goto(I_i, prod_exp[dot_pos])
                    j = itemset_collection.index(I_j)
                    table_action[i, ~a] = j << 2 | 3
                # case: accept
                elif dot_pos == len(prod_exp) and\
                        prod_exp[0] == 0 and lasym == self.terminal_map['EOF']:
                    table_action[i, ~lasym] = 1
                # case: reduce
                elif dot_pos == len(prod_exp):
                    table_action[i, ~lasym] = prod << 2 | 2
                else:
                    pass

        for i, I_i in enumerate(itemset_collection):
            for A in range(len(self.nonterminals)):
                I_j = self.goto(I_i, A)
                try:
                    j = itemset_collection.index(I_j)
                    table_goto[i, A] = j
                except ValueError:
                    pass  # Do nothing

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

    def print_analysis_table(self, terminal_formatter=lambda x: x):
        if self._table_action is None or self._table_goto is None:
            return

        size_state, size_action = self._table_action.shape
        _, size_goto = self._table_goto.shape

        def str_val(n):
            if n & 3 == 0:
                return ''
            elif n & 3 == 1:
                return 'acc'
            elif n & 3 == 2:
                return 'r' + str(n >> 2)
            elif n & 3 == 3:
                return 's' + str(n >> 2)

        table_str_list = []

        def format_terminal(t):
            if not getattr(t, '_augmented', False):
                t = terminal_formatter(t)
            return f'{t:<8}'

        table_header_str = '    | '
        table_header_str += ' '.join(map(format_terminal, self.terminals))
        table_header_str += ' | '
        table_header_str += ' '.join(f'{x:<8}' for x in self.nonterminals)
        table_str_list.append('-' * len(table_header_str))
        table_str_list.append(table_header_str)
        table_str_list.append('-' * len(table_header_str))

        for i in range(size_state):
            table_body_str = f'{i:>3} | '
            table_body_str += ' '.join(
                '{:<8}'.format(str_val(self._table_action[i, j]))
                for j in range(size_action))
            table_body_str += ' | '
            table_body_str += ' '.join(
                '{:<8}'.format(self._table_goto[i, j] or '')
                for j in range(size_goto))
            table_str_list.append(table_body_str)

        table_str_list.append('-' * len(table_header_str))

        final_str = '\n'.join(table_str_list)
        print(final_str)


