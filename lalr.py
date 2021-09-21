from com import AUG_SYMBOL_EOF, GrammarError


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
    def _internal_first(self, is_beta, s):
        trace = []

        def first_single(X):
            if X < 0:
                return [X]

            fst = []
            for prod in self._prod_map[X]:
                prod_exp = self.prods[prod]
                if len(prod_exp) == 1 or prod_exp[1] is None:
                    fst.append(None)
                else:
                    trace.append(X)
                    fst += first_sequence(prod_exp[1:])
                    trace.pop()
            return fst

        def first_sequence(beta):
            fst = []
            for X in beta:
                if X in trace:  # avoid production loop
                    break
                X_first = first_single(X)
                fst += (s for s in X_first if s is not None)
                if None not in X_first:
                    break
            else:
                fst.append(None)
            return fst

        if is_beta:
            return first_sequence(s)
        else:
            return first_single(s)

    def first(self, X, lookup_cache=True):
        if lookup_cache and X in self._first_map:
            return self._first_map[X]

        fst = self._first_map[X] = self._internal_first(False, X)
        return fst

    def first_beta(self, beta):
        return self._internal_first(True, beta)

    # ---------------
    # Routines used to construct LR(0) itemset collection
    # ---------------
    def closure_lr0(self, I):
        J = I[:]
        mark = set()  # mark productions had been added to J

        v, w = 0, len(J)
        while v < w:
            prod_idx, dot_pos = J[v]
            prod_exp = self.prods[prod_idx]
            if dot_pos < len(prod_exp) and prod_exp[dot_pos] >= 0:
                for p_idx in self._prod_map[prod_exp[dot_pos]]:
                    if p_idx not in mark:
                        J.append((p_idx, 1))
                        w += 1
                        mark.add(p_idx)
            v += 1

        J.sort()
        return J

    def goto_lr0(self, I, X, *, kernel=False):
        cache_key = (id(I), X)  # TODO
        if cache_key in self._goto_cache:
            return self._goto_cache[cache_key]

        J = []
        for prod_idx, dot_pos in I:
            prod_exp = self.prods[prod_idx]
            if dot_pos < len(prod_exp) and prod_exp[dot_pos] == X:
                J.append((prod_idx, dot_pos + 1))

        self._goto_cache[cache_key] = JJ = self.closure_lr0(J)
        if kernel:
            return J, JJ
        else:
            return JJ

    def items_lr0(self):
        D = [[(0, 1)]]
        C = [self.closure_lr0(k) for k in D]
        goto_track = [{}]

        v, w = 0, len(C)
        while v < w:
            I = C[v]

            # pick out symbols following dot
            interesting_symbols = set()
            for prod_idx, dot_pos in I:
                prod_exp = self.prods[prod_idx]
                if dot_pos < len(prod_exp):
                    interesting_symbols.add(prod_exp[dot_pos])

            for X in interesting_symbols:
                gK, gI = self.goto_lr0(I, X, kernel=True)
                if len(gI) == 0:
                    continue
                try:
                    gI_idx = C.index(gI)
                except Exception:
                    C.append(gI)
                    D.append(gK)
                    goto_track.append({})
                    goto_track[v][X] = len(C) - 1
                    w += 1
                else:
                    goto_track[v][X] = gI_idx

            v += 1

        self.kernel_collection = D
        self.itemset_collection = C
        self.goto_track = goto_track

    # ----------------
    # Routines for LALR grammar
    # ----------------
    def lalr_closure(self, I):
        I = I[:]
        existed_item = {}

        v, w = 0, len(I)
        while v < w:
            prod, dot_pos, las = I[v]
            prod_exp = self.prods[prod]
            if dot_pos < len(prod_exp):
                if (dot_sym := prod_exp[dot_pos]) >= 0:
                    # compute frist symbols as two parts:
                    #   producted symbols at right of dot and lookahead symbols
                    right_fst = self.first_beta(prod_exp[dot_pos+1:])
                    total_fst = las[:] if None in right_fst else []
                    for t in right_fst:
                        if t is not None and t not in total_fst:
                            total_fst.append(t)
                    # stuff the lookahead list of lalr item
                    for y in self._prod_map[dot_sym]:
                        if y in existed_item:
                            existed_las = I[existed_item[y]][2]
                            for t in total_fst:
                                if t not in existed_las:
                                    existed_las.append(t)
                        else:
                            I.append((y, 1, total_fst))
                            w += 1
                            existed_item[y] = w - 1
            v += 1

        for item in I:
            item[2].sort(reverse=True)
        I.sort(key=lambda item: item[0:1])
        return I

    # -----------------
    # Construct LALR itemset by attaching lookahead symbols to LR(0) itemset
    # -----------------
    def discover_lookahead(self):
        SHARP_SYMBOL = self.terminal_map['propagate_placeholder']
        EOF_SYMBOL = self.terminal_map['EOF']

        # initialize tables
        propagate_table = self.lookahead_propagate_table\
            = [[[] for _ in K] for K in self.kernel_collection]
        kernel_collection = self.lalr_kernel_collection\
            = [[(*kitem, []) for kitem in K] for K in self.kernel_collection]
        for kitem in kernel_collection[0]:
            kitem[2].append(EOF_SYMBOL)

        for K_idx, K in enumerate(kernel_collection):
            for ki, (k_prod, k_dot_pos, _) in enumerate(K):
                J = self.lalr_closure([(k_prod, k_dot_pos, [SHARP_SYMBOL])])
                for prod, dot_pos, las in J:
                    prod_exp = self.prods[prod]
                    if dot_pos < len(prod_exp):
                        X = prod_exp[dot_pos]
                        p_kernel = self.goto_track[K_idx][X]
                        p_item = next(idx for idx, e in enumerate(kernel_collection[p_kernel])
                                      if e[0] == prod and e[1] == dot_pos+1)

                        if SHARP_SYMBOL in las:
                            propagate_table[K_idx][ki].append((p_kernel, p_item))

                        p_las = kernel_collection[p_kernel][p_item][2]
                        for s in las:
                            if s is not None and s != SHARP_SYMBOL and s not in p_las:
                                p_las.append(s)

    def propagate_lookahead(self):
        propagate_table = self.lookahead_propagate_table
        kernel_collection = self.lalr_kernel_collection

        cnt = 1  # trace how many symbols are propagated in a loop turn
        while cnt:
            cnt = 0
            for source_kernel, propagate_table_kernel in enumerate(propagate_table):
                for source_item, propagate_table_kernel_item in enumerate(propagate_table_kernel):
                    for target_kernel, target_item in propagate_table_kernel_item:
                        source_lookahead_list = kernel_collection[source_kernel][source_item][2]
                        target_lookahead_list = kernel_collection[target_kernel][target_item][2]
                        for s in source_lookahead_list:
                            if s not in target_lookahead_list:
                                target_lookahead_list.append(s)
                                cnt += 1

    def lalr_items(self):
        self.lalr_itemset_collection\
            = list(map(self.lalr_closure, self.lalr_kernel_collection))

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
                    j = self.goto_track[i][a]
                    set_action(i, a, j << 2 | 3, a)
                # case 2) accept
                elif prod_exp[0] == 0 and dot_pos == len(prod_exp):
                    a = self.terminal_map['EOF']
                    set_action(i, a, 1, a)
                # case 3) reduce
                elif dot_pos == len(prod_exp):
                    for a in las:
                        set_action(i, a, prod_idx << 2 | 2, a)
                # case 4) error
                else:
                    pass  # Each cell is set with "0" by default.

        # Construct GOTO table for each state
        for i in range(len(self.lalr_itemset_collection)):
            for A, j in self.goto_track[i].items():
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
            + '/'.join(str(self.terminals[~t]) for t in las)

    def print_lr0_itemset_collection(self):
        for i, itemset in enumerate(self.itemset_collection):
            print(f'C[{i}]')
            print(*('    ' + self.stringify_lr0_item(t) for t in itemset), sep='\n')

    def print_lr0_kernel_collection(self):
        for i, itemset in enumerate(self.kernel_collection):
            print(f'K[{i}]')
            print(*('    ' + self.stringify_lr0_item(t) for t in itemset), sep='\n')

    def print_lookahead_propagate_table(self):
        table = self.lookahead_propagate_table
        kernel_collection = self.kernel_collection
        for K_i, K in enumerate(table):
            for ki, propagate_targets in enumerate(K):
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


