import time
import cProfile as profile
from com import AUG_SYMBOL_EOF


class GrammarError(Exception):
    pass


class GrammarBase():
    pass


class GrammarSlr(GrammarBase):
    def __init__(self):
        self.prods = []
        self.symbols = []
        self.terminal_symbols = []
        self.nonterminal_symbols = []
        self.start_symbol = None

        self._prod_index = {}
        self._first_map = {}
        self._follow_map = {}
        self._itemcol = []
        self._table_goto = None
        self._table_action = None

    def set_grammar(self, *, prods=None,
                    terminal_symbols=None, nonterminal_symbols=None,
                    precedence_map=None,
                    start_symbol=None,
                    abs_prods=None):
        prods = list(prods) if prods else []
        terminal_symbols = list(terminal_symbols) if terminal_symbols else []
        if AUG_SYMBOL_EOF not in terminal_symbols:  # TODO
            terminal_symbols.insert(0, AUG_SYMBOL_EOF)
        nonterminal_symbols = list(nonterminal_symbols) if nonterminal_symbols else []
        precedence_map = {} if precedence_map is None else precedence_map

        self.prods = prods
        self.terminal_symbols = terminal_symbols
        self.nonterminal_symbols = nonterminal_symbols
        self.symbols = nonterminal_symbols + terminal_symbols
        self.start_symbol = start_symbol if start_symbol else nonterminal_symbols[0]
        self.precedence_map = precedence_map
        self.abs_prods = abs_prods

        self._compute_prod_index()

    def _first(self, X, lookup_cache=True, trace=None):
        if lookup_cache and X in self._first_map:
            return self._first_map[X]

        if trace is None:
            trace = []

        if X < 0:
            fst = self._first_map[X] = [X]
            return fst

        # X is an nonterminal symbol
        fst = []
        for p in (p for p in self.abs_prods if p[0] == X):
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
                # TODO: Consider the following productions:
                #   exp -> exp RIGHT
                #   exp -> None
            X_first = self._first(X, trace=trace)
            ret += (s for s in X_first if s is not None)
            if None not in X_first:
                break
        else:
            ret.append(None)
        return ret

    def _compute_follows(self):
        flws = self._follow_map = [[] for _ in self.nonterminal_symbols]
        flws[0].append(~self.terminal_symbols.index(AUG_SYMBOL_EOF))

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
            for prod_exp in self.abs_prods:
                A = prod_exp[0]
                if len(prod_exp) == 1 or prod_exp[1] is None:
                    continue
                for i in range(1, len(prod_exp)):
                    B = prod_exp[i]
                    if not B >= 0:
                        continue
                    if i == len(prod_exp) - 1:
                        some_added += unique_merge(flws[A], flws[B])
                    else:
                        beta_first = self._first_beta(prod_exp[i+1:])
                        some_added += unique_merge(
                            (s for s in beta_first if s is not None), flws[B])
                        if None in beta_first:
                            some_added += unique_merge(flws[A], flws[B])
        return flws

    def _compute_prod_index(self):
        # TODO: How about use array indexed with symbol number?
        pindex = self._prod_index = {}
        for prod_idx, prod_exp in enumerate(self.prods):
            if (left := prod_exp[0]) not in pindex:
                lst = pindex[left] = []
            else:
                lst = pindex[left]
            lst.append(prod_idx)

        pindex = self._abs_prod_map = [[] for _ in range(len(self.nonterminal_symbols))]
        for abs_prod_idx, abs_prod_exp in enumerate(self.abs_prods):
            pindex[abs_prod_exp[0]].append(abs_prod_idx)

    def closure(self, I):  # noqa: E741
        J = I[:]

        # Mark whether (prod_idx, 1) has existed.
        new_item_mark = bytearray(len(self.prods) // 8 + 1)
        for prod_idx, dot_pos in J:
            if dot_pos == 1:
                new_item_mark[prod_idx//8] |= 1 << prod_idx % 8

        # Loop to extend I
        v, w = 0, len(J)
        while v < w:
            prod_idx, dot_pos = J[v]
            prod_exp = self.abs_prods[prod_idx]
            if dot_pos < len(prod_exp) and prod_exp[dot_pos] >= 0:
                for p_idx in self._abs_prod_map[prod_exp[dot_pos]]:
                    if not new_item_mark[p_idx//8] & (1 << p_idx % 8):
                        J.append((p_idx, 1))
                        w += 1
                        new_item_mark[p_idx//8] |= 1 << p_idx % 8
            v += 1

        J.sort()
        return J

    def goto(self, I, X):  # noqa: E741
        J = []
        for prod_idx, dot_pos in I:
            prod_exp = self.abs_prods[prod_idx]
            if dot_pos < len(prod_exp) and prod_exp[dot_pos] == X:
                J.append((prod_idx, dot_pos + 1))
        return self.closure(J)

    def _compute_itemcol(self):
        C = [self.closure([(0, 1)])]
        v, w = 0, len(C)
        while v < w:
            I = C[v]  # noqa: E741

            # we care no symbols but following symbols.
            following_symbols = []
            for prod_idx, dot_pos in I:
                prod_exp = self.abs_prods[prod_idx]
                if dot_pos < len(prod_exp):
                    if (s := prod_exp[dot_pos]) not in following_symbols:
                        following_symbols.append(s)

            for X in following_symbols:
                g = self.goto(I, X)
                if len(g) > 0 and g not in C:
                    C.append(g)
                    w += 1

            v += 1
        return C

    def generate_analysis_table(self):
        number_of_state = len(self._itemcol)
        number_of_nonterminal_symbols = len(self.nonterminal_symbols)
        number_of_terminal_symbols = len(self.terminal_symbols)

        # Initialize table storage space.
        x, y = number_of_state, number_of_terminal_symbols
        self._table_action = memoryview(bytearray(x * y * 4)).cast('L', (x, y))
        x, y = number_of_state, number_of_nonterminal_symbols
        self._table_goto = memoryview(bytearray(x * y * 4)).cast('L', (x, y))

        print('AC: %s' % (time.time(), ))

        def get_precedence_key_terminal(prod):
            if prod.prec:
                return prod.prec
            for s in reversed(prod[1:]):
                if s in self.terminal_symbols:
                    return s
            else:
                return None

        # the helper routine to solve a pair of action conflicts.
        def solve_conflict(prev_action, next_action, terminal=None):
            if prev_action == next_action:
                return prev_action

            _DEFAULT_PRECEDENCE = (None, 'right', 0)

            prev_type, prev_value = (prev_action & 3), (prev_action >> 2)
            next_type, next_value = (next_action & 3), (next_action >> 2)

            # shift/shift conflict: is a grammar error
            if prev_type == 3 and next_type == 3:
                raise GrammarError('Unsolvable grammar conflict (shift/shift).')

            # reduce/reduce conflict: use the first defined production
            elif prev_type == 2 and next_type == 2:
                return prev_action if prev_value < next_value else next_action

            # reduce/shift conflict: compare precedence
            elif prev_type == 2 and next_type == 3:
                _, prev_assc, prev_level = self.precedence_map.get(
                    get_precedence_key_terminal(self.prods[prev_value]), _DEFAULT_PRECEDENCE)
                _, next_assc, next_level = self.precedence_map.get(terminal, _DEFAULT_PRECEDENCE)
                if prev_level == next_level:  # assert r_assc == s_assc
                    # two items with the same level always have the same assoc
                    if next_assc == 'left':
                        return prev_action
                    elif next_assc == 'right':
                        return next_action
                    elif next_assc == 'nonassoc':
                        return 0  # TODO: syntax error
                else:
                    return prev_action if prev_level > next_level else next_action

            # shift/reduce conflict: compare precedence
            elif prev_type == 3 and next_type == 2:
                _, prev_assc, prev_level = self.precedence_map.get(terminal, _DEFAULT_PRECEDENCE)
                _, next_assc, next_level = self.precedence_map.get(
                    get_precedence_key_terminal(self.prods[next_value]), _DEFAULT_PRECEDENCE)
                if prev_level == next_level:  # assert r_assc == s_assc
                    if next_assc == 'left':
                        return next_action
                    elif next_assc == 'right':
                        return prev_action
                    elif next_assc == 'nonassoc':
                        return 0  # TODO: syntax error
                else:
                    return prev_action if prev_level > next_level else next_action

            # any other conflict is unsolvable
            else:
                raise GrammarError('Unsolvable grammar conflict (*/*).')

        def update_cell_action(prev_action, next_action, coming_terminal):
            if prev_action == 0:
                return next_action
            return solve_conflict(prev_action, next_action,
                                 self.terminal_symbols[~coming_terminal])

        # Set ACTION table for each state
        for i, itemset_i in enumerate(self._itemcol):
            for prod_idx, dot_pos in itemset_i:
                prod_exp = self.abs_prods[prod_idx]

                # case 1) shift
                if dot_pos < len(prod_exp) and prod_exp[dot_pos] < 0:
                    a = prod_exp[dot_pos]
                    itemset_j = self.goto(itemset_i, a)
                    j = self._itemcol.index(itemset_j)  # TODO: accelerate with hash
                    self._table_action[i, ~a] =\
                        update_cell_action(self._table_action[i, ~a], j << 2 | 3, a)

                # case 2) accept
                elif prod_exp[0] == 0 and dot_pos == len(prod_exp):
                    a = ~self.terminal_symbols.index(AUG_SYMBOL_EOF)
                    self._table_action[i, ~a] =\
                        update_cell_action(self._table_action[i, ~a], 1, a)

                # case 3) reduce
                elif dot_pos == len(prod_exp):
                    for a in self._follow_map[prod_exp[0]]:
                        self._table_action[i, ~a] =\
                            update_cell_action(self._table_action[i, ~a], prod_idx << 2 | 2, a)

                # case 4) error
                else:
                    # Each cell is set with "0" by default.
                    pass

        print('AD: %s' % (time.time(), ))

        # Set GOTO table for each state
        for i, itemset_i in enumerate(self._itemcol):
            for A in range(len(self.nonterminal_symbols)):
                itemset_j = self.goto(itemset_i, A)
                if itemset_j in self._itemcol:
                    j = self._itemcol.index(itemset_j)
                    self._table_goto[i, A] = j
                else:
                    pass  # Do nothing

        print('AE: %s' % (time.time(), ))

    def compile(self):
        pr = profile.Profile()
        pr.enable()
        self._itemcol = self._compute_itemcol()
        pr.disable()
        pr.print_stats()
        # assert False
        if not self._follow_map:  # TODO
            self._compute_follows()
        self.generate_analysis_table()

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
        table_header_str += ' '.join(map(format_terminal, self.terminal_symbols))
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

