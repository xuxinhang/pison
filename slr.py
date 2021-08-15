from enum import Enum


def _is_item_collection_equal(A, B):
    if len(A) != len(B):
        return False
    for item in A:
        if item not in B:
            return False
    return True


class Ns(Enum):
    EE = 1
    E = 2
    T = 3
    F = 4


class Ts(Enum):
    PLUS = 3
    STAR = 4
    LP = 5
    RP = 6
    ID = 7


PRESET_PRODUCTIONS = [
    (Ns.EE, Ns.E),
    (Ns.E, Ns.E, Ts.PLUS, Ns.T), (Ns.E, Ns.T),
    (Ns.T, Ns.T, Ts.STAR, Ns.F), (Ns.T, Ns.F),
    (Ns.F, Ts.LP, Ns.E, Ts.RP), (Ns.F, Ts.ID),
]


class GrammarBox(object):
    def __init__(self):
        self.prods = PRESET_PRODUCTIONS
        self.symbols = list(Ns) + list(Ts)
        self.terminal_symbols = [0] + list(iter(Ts))
        self.nonterminal_symbols = list(iter(Ns))

        self._first_cache = {}
        self._follow_cache = {}
        self._item_collections_cache = []

        self._table_goto_buffer = None
        self._table_action_buffer = None
        self._table_goto = None
        self._table_action = None

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
        flws[Ns.EE].append(0)

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

    def closure(self, I):
        J = I.copy()
        flag = False
        while True:
            flag = False
            for prod_i, dot_i in J:
                prod = self.prods[prod_i]
                if dot_i >= len(prod) or prod[dot_i] not in Ns:
                    continue
                after_symbol = prod[dot_i]
                for prod_i, prod in enumerate(self.prods):
                    if prod[0] != after_symbol:
                        continue
                    new_item = (prod_i, 1)
                    if new_item not in J:
                        J.append(new_item)
                        flag = True
            if flag is False:
                break

        J.sort()
        return J

    def goto(self, I, X):
        J = []
        for item_prod_id, item_dot_pos in I:
            item_prod_exp = self.prods[item_prod_id]
            if item_dot_pos < len(item_prod_exp) and\
                    item_prod_exp[item_dot_pos] == X:
                J.append((item_prod_id, item_dot_pos + 1))
        return self.closure(J)

    def items(self):
        C = [self.closure([(0, 1)])]
        has_new_collection_added = True
        while has_new_collection_added:
            has_new_collection_added = False
            for k in range(len(C)):
                I = C[k]
                for X in self.symbols:
                    g = self.goto(I, X)
                    if len(g) > 0 and g not in C:
                        C.append(g)
                        has_new_collection_added = True
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
                    a_i = self.terminal_symbols.index(a)
                    item_j = self.goto(item_i, a)
                    j = self._item_collections_cache.index(item_j)  # TODO
                    print(i, a_i, a, prod_exp, self._table_action[i, a_i])
                    assert self._table_action[i, a_i] == 0
                    self._table_action[i, a_i] = j << 2 | 3
                elif prod_exp[0] == Ns.EE and dot_pos == len(prod_exp):
                    a_i = 0
                    assert self._table_action[i, a_i] == 0
                    self._table_action[i, a_i] = 1
                elif dot_pos == len(prod_exp):
                    for a in self._follow_cache[prod_exp[0]]:
                        a_i = self.terminal_symbols.index(a)
                        assert self._table_action[i, a_i] == 0
                        self._table_action[i, a_i] = prod_idx << 2 | 2

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


grammar = GrammarBox()
print('== CLOSURE ==')
print(grammar.stringify_item_collection(grammar.closure([(0, 1)])))

print('== CLOSURE ==')
print(grammar.stringify_item_collection(grammar.closure([(3, 3)])))

print('== GOTO ==')
print(grammar.stringify_item_collection(grammar.goto([(0, 2), (1, 2)], Ts.PLUS)))

print('== ITEMS ==')
for i, s in enumerate(grammar.items()):
    print(f'[{i}]')
    print(grammar.stringify_item_collection(s))

print('== FIRST ==')
for s in grammar.nonterminal_symbols:
    print(f'FIRST({s})', grammar._first(s))
# for s in grammar.terminal_symbols:
#     print(f'FIRST({s})', grammar._first(s))

print('== FOLLOW ==')
for k, f in grammar._compute_follows().items():
    print(f'FOLLOW({k})', f)

print('== LR Table ==')
grammar.generate_analysis_table()
grammar.print_analysis_table()

