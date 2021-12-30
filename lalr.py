from com import AUG_SYMBOL_EOF, GrammarError
from bitset import create_bitset, get_bit, iterate_bitset, or_bitset, set_bit


class GrammarBase():
    pass


class GrammarLalr(GrammarBase):
    def __init__(self):
        # grammar definition
        self.prods = None
        self.terminals = None
        self.nonterminals = None

        # grammar itemset
        self.itemset_collection = None
        self.kernel_collection = None

        # grammar parsing table
        self.parsing_table_goto = None
        self.parsing_table_action = None

        # helper cache
        self._goto_cache = {}

    def set_grammar(self, *,
                    productions, terminals, nonterminals,
                    precedence_map=None):
        self.prods = productions[:]
        self.terminals = terminals[:]
        self.nonterminals = nonterminals[:]
        self.precedence_map = {} if precedence_map is None else precedence_map.copy()

        # prepare useful cache
        self._prods_len = [len(_) for _ in self.prods]
        self._prod_map = [[] for _ in self.nonterminals]
        for prod_idx, prod_exp in enumerate(self.prods):
            self._prod_map[prod_exp[0]].append(prod_idx)

        self._first_map = {}

        self.terminal_map = {}
        self.terminal_map['EOF'] = ~self.terminals.index(AUG_SYMBOL_EOF)

        # attach helper symbols
        self.terminals.append('#')
        self.terminal_map['propagate_placeholder'] = ~(len(self.terminals) - 1)

    # ---------
    # Basic grammar routines
    # ---------
    def _run_first_with_trace(self, is_beta: bool, s):
        # "trace" records the path to the current symbol whose items
        # are tuple (symbol: int, maybe_empty: bool)
        trace: list[tuple[int, bool]] = []

        def first_single(X: int):
            if X < 0:  # for terminals
                return [X]

            if X in self._first_map:  # lookup cache
                return self._first_map[X]

            fst = []
            maybe_empty = False

            # First, scan all the productions to decide whether this symbol
            # may be empty.
            for prod in self._prod_map[X]:
                prod_exp = self.prods[prod]
                if len(prod_exp) == 1 or prod_exp[1] is None:
                    maybe_empty = True
            if maybe_empty:
                fst.append(None)

            # Then, process other productions.
            # Because we have known whether this symbol can be empty, so its
            # recursive productions can be processed properly.
            for prod in self._prod_map[X]:
                prod_exp = self.prods[prod]
                if len(prod_exp) == 1 or prod_exp[1] is None:
                    pass
                else:
                    trace.append((X, maybe_empty))
                    for s in first_sequence(prod_exp[1:]):
                        if s not in fst:
                            fst.append(s)
                    trace.pop()

            self._first_map[X] = fst
            return fst

        def first_sequence(beta: list[int]):
            fst = []
            for X in beta:
                maybe_epsilon = False
                # not to trap into the recursive symbols
                for trace_symbol, trace_maybe_empty in trace:
                    if trace_symbol == X:
                        maybe_epsilon = trace_maybe_empty
                        break
                else:
                    for s in first_single(X):
                        if s is None:
                            maybe_epsilon = True
                        else:
                            if s not in fst:
                                fst.append(s)
                if not maybe_epsilon:
                    break
            else:
                if None not in fst:
                    fst.append(None)
            return fst

        if is_beta:
            return first_sequence(s)
        else:
            return first_single(s)

    def first(self, X):
        return self._run_first_with_trace(False, X)

    def first_beta(self, beta):
        return self._run_first_with_trace(True, beta)

    # ---------------
    # Routines used to construct LR(0) itemset collection
    # ---------------
    def closure_lr0(self, I):
        J = I[:]

        # Prepare some cache sets to speed up
        # 1) memorize production left parts like "B" in "B → ∙γ" added to J by this routine
        added_dot_ahead_prod_left = set()
        # 2) memorize productions like "B → ∙γ" originally existed in J
        existed_dot_ahead_prod = frozenset(p for p, d in J if d == 1)

        v, w = 0, len(J)
        while v < w:
            prod_idx, dot_pos = J[v]
            prod_exp, prod_exp_len = self.prods[prod_idx], self._prods_len[prod_idx]
            if dot_pos < prod_exp_len and (dot_symbol := prod_exp[dot_pos]) >= 0:
                if dot_symbol in added_dot_ahead_prod_left:
                    pass
                else:
                    for p_idx in self._prod_map[dot_symbol]:
                        if p_idx not in existed_dot_ahead_prod:
                            J.append((p_idx, 1))
                            w += 1
                    added_dot_ahead_prod_left.add(dot_symbol)
            v += 1

        J.sort()
        return J

    def goto_lr0(self, I, X):
        J = []
        for prod_idx, dot_pos in I:
            prod_exp, prod_exp_len = self.prods[prod_idx], self._prods_len[prod_idx]
            if dot_pos < prod_exp_len and prod_exp[dot_pos] == X:
                J.append((prod_idx, dot_pos + 1))
        # No "J.sort()" here, because if I is sorted, then J must be sorted.
        return J, self.closure_lr0(J)

    def items_lr0(self):
        D = [[(0, 1)]]
        C = [self.closure_lr0(_) for _ in D]
        goto_graph = {}

        # For better search performance in itemset list D, compute and memorize
        # the features of each itemset.
        def get_feature(itemset):
            # compute the feature of the given itemset
            ans = itemset[0][0] << 24 ^ itemset[0][1] << 16\
                ^ itemset[-1][0] << 8 ^ itemset[-1][1] << 0
            sft = len(itemset) % 32
            return ans >> (32-sft) | (ans & 0xffffffff >> sft) << sft
        kernel_feature_map = {}
        for i, k in enumerate(D):
            kernel_feature_map.setdefault(get_feature(k), []).append(i)

        potential_symbols = set()
        v, w = 0, len(C)
        while v < w:
            I = C[v]

            # scan through the set of items to pick out symbols following dot
            # only those symbols can lead to no empty GOTO result.
            potential_symbols.clear()
            for prod_idx, dot_pos in I:
                prod_exp, prod_exp_len = self.prods[prod_idx], self._prods_len[prod_idx]
                if dot_pos < prod_exp_len:
                    potential_symbols.add(prod_exp[dot_pos])

            # Compute GOTO for each potential symbol
            for X in potential_symbols:
                gK, gI = self.goto_lr0(I, X)
                gK_feature = get_feature(gK)
                # the GOTO result is never empty due to only computing on potential symbols.
                #   if len(gI) == 0: continue
                # to find an existing set of items
                D_index = -1
                if feature_list := kernel_feature_map.get(gK_feature):
                    for k in feature_list:
                        if D[k] == gK:
                            D_index = k
                            break
                if D_index == -1:
                    D.append(gK)
                    C.append(gI)
                    w += 1
                    D_index = w - 1
                    kernel_feature_map.setdefault(gK_feature, []).append(D_index)
                goto_graph[(v, X)] = D_index
            v += 1

        self.kernel_collection = D
        self.itemset_collection = C
        self.goto_graph = goto_graph

    # ----------------
    # Routines for LALR grammar
    # ----------------
    def lalr_closure(self, I):
        I = I[:]
        added_dot_ahead_prod = {s[0]: i for i, s in enumerate(I) if s[1] == 1}

        v, w = 0, len(I)
        while v < w:
            prod, dot_pos, las = I[v]
            prod_exp = self.prods[prod]
            if dot_pos < len(prod_exp):
                if (dot_sym := prod_exp[dot_pos]) >= 0:
                    # 1) compute FIRST(ba) as lookahead symbols of new items
                    fst_b = self.first_beta(prod_exp[dot_pos+1:])
                    fst_ba = create_bitset(len(self.terminals))
                    for b in fst_b:
                        if b is None:
                            or_bitset(fst_ba, las)
                        else:
                            set_bit(fst_ba, ~b)
                    # 2) stuff the lookahead list of lalr item
                    for y in self._prod_map[dot_sym]:
                        if y in added_dot_ahead_prod:
                            or_bitset(I[added_dot_ahead_prod[y]][2], fst_ba)
                        else:
                            I.append((y, 1, fst_ba[:]))  # use copy
                            w += 1
                            added_dot_ahead_prod[y] = w - 1
            v += 1

        I.sort(key=lambda item: item[0:1])
        return I

    # -----------------
    # Construct LALR itemset by attaching lookahead symbols to LR(0) itemset
    # -----------------
    def discover_lookahead(self):
        EOF_SYMBOL = self.terminal_map['EOF']
        SHARP_SYMBOL = self.terminal_map['propagate_placeholder']
        ONE_HOT_SHARP_SYMBOL_BITSET = create_bitset(len(self.terminals))
        set_bit(ONE_HOT_SHARP_SYMBOL_BITSET, ~SHARP_SYMBOL)

        # initialize tables
        propagate_graph = self.lookahead_propagate_graph = {}
        kernel_collection = self.lalr_kernel_collection =\
            [[(*kitem, create_bitset(len(self.terminals))) for kitem in K]\
             for K in self.kernel_collection]
        for kitem in kernel_collection[0]:
            set_bit(kitem[2], ~EOF_SYMBOL)

        for K_idx, K in enumerate(kernel_collection):
            for ki, (k_prod, k_dot_pos, _) in enumerate(K):
                J = self.lalr_closure([(k_prod, k_dot_pos, ONE_HOT_SHARP_SYMBOL_BITSET[:])])
                for prod, dot_pos, las in J:
                    prod_exp, prod_exp_len = self.prods[prod], self._prods_len[prod]
                    if not dot_pos < prod_exp_len:
                        continue
                    X = prod_exp[dot_pos]
                    goto_kernel = self.goto_graph[(K_idx, X)]
                    for goto_item, e in enumerate(kernel_collection[goto_kernel]):
                        if e[0] == prod and e[1] == dot_pos + 1:
                            break  # We assert there must be such item.
                    # case 1:
                    or_bitset(kernel_collection[goto_kernel][goto_item][2], las)
                    # case 2:
                    if get_bit(las, ~SHARP_SYMBOL):
                        propagate_graph.setdefault((K_idx, ki), []).append((goto_kernel, goto_item))

    def propagate_lookahead(self):
        propagate_graph = self.lookahead_propagate_graph
        kernel_collection = self.lalr_kernel_collection

        changed = 1
        while changed:  # loop until there is no more change.
            changed = 0
            for (source_kernel, source_item), target_list in propagate_graph.items():
                for (target_kernel, target_item) in target_list:
                    source_lookahead_set = kernel_collection[source_kernel][source_item][2]
                    target_lookahead_set = kernel_collection[target_kernel][target_item][2]
                    prev_target_lookahead_set = bytearray(target_lookahead_set)
                    or_bitset(target_lookahead_set, source_lookahead_set)
                    changed |= (prev_target_lookahead_set != target_lookahead_set)

    def lalr_items(self):
        self.lalr_itemset_collection = [self.lalr_closure(K) for K in self.lalr_kernel_collection]

    def construct_parsing_table(self):
        # Initialize table storage space.
        def create_table(x, y):
            return memoryview(bytearray(x * y * 4)).cast('L', (x, y))

        state_number = len(self.lalr_itemset_collection)
        self.parsing_table_action = create_table(state_number, len(self.terminals))
        self.parsing_table_goto = create_table(state_number, len(self.nonterminals))

        # Construct ACTION table for each state
        table_action = self.parsing_table_action

        def get_production_precedence_terminal(prod):
            # prod_object = self._original_productions[prod]
            prod_object = self.prods[prod]  # HACK
            if prod_object.prec is not None:
                return prod_object.prec

            prod_exp = self.prods[prod]
            for s in reversed(prod_exp):
                if s < 0:
                    return self.terminals[~s]
            else:
                return None

        def solve_conflict(prev_action, next_action, coming_terminal=None):
            DEFAULT_PRECEDENCE = (None, 'right', 0)
            prev_type, prev_value = (prev_action & 3), (prev_action >> 2)
            next_type, next_value = (next_action & 3), (next_action >> 2)

            # shift/shift conflict: is a grammar error
            if prev_type == 3 and next_type == 3:
                raise GrammarError('Unsolvable grammar conflict (shift/shift)')
            # reduce/reduce conflict: use the first defined production
            elif prev_type == 2 and next_type == 2:
                return prev_action if prev_value < next_value else next_action
            # reduce/shift conflict: compare precedence
            elif prev_type == 2 and next_type == 3:
                _, prev_assc, prev_level = self.precedence_map.get(
                    get_production_precedence_terminal(prev_value), DEFAULT_PRECEDENCE)
                _, next_assc, next_level = self.precedence_map.get(
                    self.terminals[~coming_terminal], DEFAULT_PRECEDENCE)
                if prev_level > next_level:
                    return prev_action
                elif prev_level < next_level:
                    return next_action
                else:  # two items with the same level always have the same assoc
                    return prev_action if next_assc == 'left' else\
                        next_action if next_assc == 'right' else 0
            # shift/reduce conflict: compare precedence
            elif prev_type == 3 and next_type == 2:
                _, prev_assc, prev_level = self.precedence_map.get(
                    self.terminals[~coming_terminal], DEFAULT_PRECEDENCE)
                _, next_assc, next_level = self.precedence_map.get(
                    get_production_precedence_terminal(next_value), DEFAULT_PRECEDENCE)
                if prev_level > next_level:
                    return prev_action
                elif prev_level < next_level:
                    return next_action
                else:
                    return next_action if next_assc == 'left' else\
                        prev_action if next_assc == 'right' else 0
            # any other conflict is unsolvable
            else:
                raise GrammarError('Unsolvable grammar conflict (*/*)')

        def set_action(i, a, next_action, coming_terminal):
            prev_action = table_action[i, ~a]
            if prev_action == next_action:
                return
            if prev_action == 0:
                table_action[i, ~a] = next_action
                return
            # there is a conflict between the old and the new actions
            table_action[i, ~a] = solve_conflict(prev_action, next_action, coming_terminal)

        for i, I in enumerate(self.lalr_itemset_collection):
            for prod_idx, dot_pos, las in I:
                prod_exp = self.prods[prod_idx]
                # case 1) shift
                if dot_pos < len(prod_exp) and prod_exp[dot_pos] < 0:
                    a = prod_exp[dot_pos]
                    j = self.goto_graph[(i, a)]
                    set_action(i, a, j << 2 | 3, a)
                # case 2) accept
                elif prod_exp[0] == 0 and dot_pos == len(prod_exp):
                    a = self.terminal_map['EOF']
                    set_action(i, a, 1, a)
                # case 3) reduce
                elif dot_pos == len(prod_exp):
                    for a in iterate_bitset(las):
                        set_action(i, ~a, prod_idx << 2 | 2, ~a)
                # case 4) error
                else:
                    pass  # Each cell is set with "0" by default.

        # Construct GOTO table for each state
        for (i, A), j in self.goto_graph.items():
            if A >= 0:
                self.parsing_table_goto[i, A] = j

    def compile(self):
        self.items_lr0()
        self.discover_lookahead()
        self.propagate_lookahead()
        self.lalr_items()
        self.construct_parsing_table()

    # ---------
    # DEBUG: format printer
    # ---------
    def stringify_production(self, prod_exp, dot_pos):
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

        return ''.join(ss)

    def stringify_lr0_item(self, item):
        prod, dot_pos = item
        prod_exp = self.prods[prod]
        return self.stringify_production(prod_exp, dot_pos)

    def stringify_lalr_item(self, item):
        prod, dot_pos, las = item
        prod_exp = self.prods[prod]
        return self.stringify_production(prod_exp, dot_pos) + ' , '\
            + '/'.join(str(self.terminals[t]) for t in iterate_bitset(las)
                       if ~t != self.terminal_map['propagate_placeholder'])

    def print_lr0_itemset_collection(self):
        for i, itemset in enumerate(self.itemset_collection):
            print(f'C[{i}]')
            print(*('    ' + self.stringify_lr0_item(t) for t in itemset), sep='\n')

    def print_lr0_kernel_collection(self):
        for i, itemset in enumerate(self.kernel_collection):
            print(f'K[{i}]')
            print(*('    ' + self.stringify_lr0_item(t) for t in itemset), sep='\n')

    def print_lookahead_propagate_table(self):
        table = self.lookahead_propagate_graph
        kernel_collection = self.kernel_collection
        for (K_i, ki), propagate_targets in table.items():
            if len(propagate_targets) == 0:
                continue
            source_item = kernel_collection[K_i][ki]
            print('I%d:  %s' % (K_i, self.stringify_lr0_item(source_item)))
            for target in propagate_targets:
                target_item = kernel_collection[target[0]][target[1]]
                print('        I%d:  %s' % (target[0], self.stringify_lr0_item(target_item)))

    def print_lookahead_generate_table(self):
        table = self.lookahead_generate_table
        kernel_collection = self.kernel_collection
        for K, table_K in enumerate(table):
            for item, table_K_item in enumerate(table_K):
                print('I%d: %s : %s' % (K,
                                        self.stringify_lr0_item(kernel_collection[K][item]),
                                        ' / '.join(str(self.terminals[~s]) for s in table_K_item)))

    def print_lalr_kernel_collection(self):
        for i, itemset in enumerate(self.lalr_kernel_collection):
            print(f'K[{i}]')
            print(*('    ' + self.stringify_lalr_item(t) for t in itemset), sep='\n')

    def print_lalr_itemset_collection(self):
        for i, itemset in enumerate(self.lalr_itemset_collection):
            print(f'C[{i}]')
            print(*('    ' + self.stringify_lalr_item(t) for t in itemset), sep='\n')

    def print_parsing_table(self, terminal_formatter=lambda x: x):
        size_state, size_action = self.parsing_table_action.shape
        _, size_goto = self.parsing_table_goto.shape

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
                '{:<8}'.format(str_val(self.parsing_table_action[i, j]))
                for j in range(size_action))
            table_body_str += ' | '
            table_body_str += ' '.join(
                '{:<8}'.format(self.parsing_table_goto[i, j] or '')
                for j in range(size_goto))
            table_str_list.append(table_body_str)

        table_str_list.append('-' * len(table_header_str))

        print('\n'.join(table_str_list))


