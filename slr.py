
class HelperSymbol:
    def __init__(self, desc):
        self.desc = desc

    def __repr__(self):
        return self.desc.__repr__()

    def __str__(self):
        return self.desc.__str__()

    def __format__(self, *args, **kwargs):
        return self.desc.__format__(*args, **kwargs)


SYMBOL_HELPER_EOF = HelperSymbol('$end')
# SYMBOL_HELPER_S = HelperSymbol('S\'')


class GrammarBase(object):
    pass


class GrammarSlr(GrammarBase):
    def __init__(self):
        self.prods = []
        self.symbols = []
        self.terminal_symbols = []
        self.nonterminal_symbols = []
        self.start_symbol = None

        self._first_cache = {}
        self._follow_cache = {}
        self._item_collections_cache = []

        self._table_goto_buffer = None
        self._table_action_buffer = None
        self._table_goto = None
        self._table_action = None

    def set_grammar(self, *, prods=None,
                    terminal_symbols=None, nonterminal_symbols=None,
                    start_symbol=None):
        prods = list(prods) if prods else []
        terminal_symbols = list(terminal_symbols) if terminal_symbols else []
        nonterminal_symbols = list(nonterminal_symbols) if nonterminal_symbols else []
        terminal_symbols.insert(0, SYMBOL_HELPER_EOF)

        self.prods = prods
        self.terminal_symbols = terminal_symbols
        self.nonterminal_symbols = nonterminal_symbols
        self.symbols = nonterminal_symbols + terminal_symbols
        self.start_symbol = start_symbol if start_symbol else nonterminal_symbols[0]

    def _first(self, X, lookup_cache=True):
        if lookup_cache and X in self._first_cache:
            return self._first_cache[X]

        if X in self.terminal_symbols:
            fst = self._first_cache[X] = [X]
            return fst

        # X is an nonterminal symbol
        fst = []
        for p in (p for p in self.prods if p[0] == X):
            # for empty productions
            if len(p) == 1 or p[1] is None:
                fst.append(None)
                continue
            # Remember to skip left-recursive productions
            fst += self._first_beta(p[1:], left=X)

        self._first_cache[X] = fst
        return fst

    def _first_beta(self, beta, left=None):
        fst = []
        for X in beta:
            if X == left:
                break
                # TODO: Consider the following productions:
                #   exp -> exp RIGHT
                #   exp -> None
            symbol_first = self._first(X)
            fst += (x for x in symbol_first if x is not None)
            if None not in symbol_first:
                break
        else:
            fst.append(None)

        return fst

    def _compute_follows(self):
        self._follow_cache = flws = {s: [] for s in self.nonterminal_symbols}
        flws[self.start_symbol].append(SYMBOL_HELPER_EOF)

        def unique_merge(fr, to):
            count = 0
            for s in fr:
                if s not in to:
                    to.append(s)
                    count += 1
            return count

        some_added = 1
        while some_added:
            some_added = 0
            for p in self.prods:
                A = p[0]
                if len(p) == 1 or p[1] is None:
                    continue
                for i in range(1, len(p)):
                    B = p[i]
                    if B not in self.nonterminal_symbols:
                        continue
                    if i == len(p) - 1:
                        some_added += unique_merge(flws[A], flws[B])
                    else:
                        beta_first = self._first_beta(p[i+1:])
                        some_added += unique_merge(
                            (s for s in beta_first if s is not None), flws[B])
                        if None in beta_first:
                            some_added += unique_merge(flws[A], flws[B])
        return flws

    def closure(self, I):  # noqa: E741
        J = I[:]

        some_added = True
        while some_added:
            some_added = False
            for prod_idx, dot_pos in J:
                prod_exp = self.prods[prod_idx]
                if dot_pos >= len(prod_exp) or prod_exp[dot_pos] not in self.nonterminal_symbols:
                    continue
                after_symbol = prod_exp[dot_pos]
                for prod_i, prod in enumerate(self.prods):
                    if prod[0] != after_symbol:
                        continue
                    new_item = (prod_i, 1)
                    if new_item not in J:
                        J.append(new_item)
                        some_added = True

        J.sort()
        return J

    def goto(self, I, X):  # noqa: E741
        J = []
        for prod_idx, dot_pos in I:
            prod_exp = self.prods[prod_idx]
            if dot_pos < len(prod_exp) and prod_exp[dot_pos] == X:
                J.append((prod_idx, dot_pos + 1))
        return self.closure(J)

    def items(self):
        C = [self.closure([(0, 1)])]
        some_added = True
        while some_added:
            some_added = False
            for I in C:  # noqa: E741
                for X in self.symbols:
                    g = self.goto(I, X)
                    if len(g) > 0 and g not in C:
                        C.append(g)
                        some_added = True
        return C

    def generate_analysis_table(self):
        self._item_collections_cache = self.items()

        number_of_state = len(self._item_collections_cache)
        number_of_nonterminal_symbols = len(self.nonterminal_symbols)
        number_of_terminal_symbols = len(self.terminal_symbols)

        self._table_action_buffer = bytearray(
            4 * number_of_state * number_of_terminal_symbols)
        self._table_action = memoryview(self._table_action_buffer)\
            .cast('L', (number_of_state, number_of_terminal_symbols))

        self._table_goto_buffer = bytearray(
            4 * number_of_state * number_of_nonterminal_symbols)
        self._table_goto = memoryview(self._table_goto_buffer)\
            .cast('L', (number_of_state, number_of_nonterminal_symbols))

        # Set ACTION table for each state
        for i, item_i in enumerate(self._item_collections_cache):
            for prod_idx, dot_pos in item_i:
                prod_exp = self.prods[prod_idx]
                if dot_pos < len(prod_exp) and prod_exp[dot_pos] in self.terminal_symbols:
                    a = prod_exp[dot_pos]
                    a_idx = self.terminal_symbols.index(a)
                    item_j = self.goto(item_i, a)
                    j = self._item_collections_cache.index(item_j)  # TODO
                    print(i, a_idx, a, prod_exp, self._table_action[i, a_idx])
                    assert self._table_action[i, a_idx] == 0
                    self._table_action[i, a_idx] = j << 2 | 3
                elif prod_exp[0] == self.start_symbol and dot_pos == len(prod_exp):
                    a_idx = 0
                    assert self._table_action[i, a_idx] == 0
                    self._table_action[i, a_idx] = 1
                elif dot_pos == len(prod_exp):
                    for a in self._follow_cache[prod_exp[0]]:
                        a_idx = self.terminal_symbols.index(a)
                        assert self._table_action[i, a_idx] == 0
                        self._table_action[i, a_idx] = prod_idx << 2 | 2

        # Set GOTO table for each state
        for i, item_i in enumerate(self._item_collections_cache):
            for A_idx, A in enumerate(self.nonterminal_symbols):
                item_j = self.goto(item_i, A)
                if item_j in self._item_collections_cache:
                    j = self._item_collections_cache.index(item_j)
                    self._table_goto[i, A_idx] = j
                else:
                    pass  # Do nothing

    def stringify_item_collection(self, item_collection):
        ret = []
        for item_prod_idx, item_dot in item_collection:
            item_prod = self.prods[item_prod_idx]
            ss = []
            ss.append(f'<{item_prod[0]:s}> -> ')
            for i, r in enumerate(item_prod[1:]):
                ss.append(('\u25AA' if i + 1 == item_dot else ' ') + f'<{r}>')
            ret.append(''.join(ss))
        return '\n'.join(ret)

    def print_analysis_table(self):
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

        table_header_str = '    | '
        table_header_str += ' '.join(f'{x:<8}' for x in self.terminal_symbols)
        table_header_str += ' | '
        table_header_str += ' '.join(f'{x:<8}' for x in self.nonterminal_symbols)
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

