from com import AUG_SYMBOL_EOF


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
        self.lr1_itemset_collection = None

        # grammar parsing table
        self.parsing_table_goto = None
        self.parsing_table_action = None

        # helper cache
        self._goto_cache = {}

    def set_grammar(self, *, productions, terminals, nonterminals):
        self.prods = productions[:]
        self.terminals = terminals[:]
        self.nonterminals = nonterminals[:]

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

        # Mark whether (prod_idx, 1) has existed.
        new_item_mark = bytearray(len(self.prods) // 8 + 1)
        for prod_idx, dot_pos in J:
            if dot_pos == 1:
                new_item_mark[prod_idx//8] |= 1 << prod_idx % 8

        # Loop to extend I
        v, w = 0, len(J)
        while v < w:
            prod_idx, dot_pos = J[v]
            prod_exp = self.prods[prod_idx]
            if dot_pos < len(prod_exp) and prod_exp[dot_pos] >= 0:
                for p_idx in self._prod_map[prod_exp[dot_pos]]:
                    if not new_item_mark[p_idx//8] & (1 << p_idx % 8):
                        J.append((p_idx, 1))
                        w += 1
                        new_item_mark[p_idx//8] |= 1 << p_idx % 8
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

    # -----------------
    # Construct LALR itemset by attaching lookahead symbols to LR(0) itemset
    # -----------------
    def discover_lookahead(self):
        goto_track = self.goto_track
        kernel_collection = self.kernel_collection
        propagate_placeholder = self.terminal_map['propagate_placeholder']

        # initialize tables
        propagate_table = self.lookahead_propagate_table\
            = [[[] for _ in K] for K in kernel_collection]
        generate_table = self.lookahead_generate_table\
            = [[[] for _ in K] for K in kernel_collection]
        for item in generate_table[0]:
            item.append(self.terminal_map['EOF'])

        for K_idx, K in enumerate(kernel_collection):
            for ki, (k_prod, k_dot_pos) in enumerate(K):
                J = self.closure([(k_prod, k_dot_pos, propagate_placeholder)])
                for prod, dot_pos, a in J:
                    prod_exp = self.prods[prod]
                    if dot_pos >= len(prod_exp):
                        continue
                    X = prod_exp[dot_pos]
                    target_kernel = goto_track[K_idx][X]
                    target_item = kernel_collection[target_kernel].index((prod, dot_pos+1))
                    if a == propagate_placeholder:
                        propagate_table[K_idx][ki].append((target_kernel, target_item))
                    else:
                        generate_table[target_kernel][target_item].append(a)

    def propagate_lookahead(self):
        generate_table = self.lookahead_generate_table
        propagate_table = self.lookahead_propagate_table

        cnt = 1  # trace how many symbols are propagated in a loop turn
        while cnt:
            cnt = 0
            for source_kernel, propagate_table_kernel in enumerate(propagate_table):
                for source_item, propagate_table_kernel_item in enumerate(propagate_table_kernel):
                    for target_kernel, target_item in propagate_table_kernel_item:
                        source_lookahead_list = generate_table[source_kernel][source_item]
                        target_lookahead_list = generate_table[target_kernel][target_item]
                        for s in source_lookahead_list:
                            if s not in target_lookahead_list:
                                target_lookahead_list.append(s)
                                cnt += 1

    # ----------------
    # Routines for LR(1) grammar
    # ----------------
    def closure(self, I):
        I = I[:]

        v, w = 0, len(I)
        while v < w:
            prod, dot_pos, lasym = I[v]
            prod_exp = self.prods[prod]
            if dot_pos < len(prod_exp):
                firsts = self.first_beta((*prod_exp[dot_pos+1:], lasym))
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

    def lr1_items(self):
        lr1_itemset_collection = []

        for kernel, lasyms in zip(self.kernel_collection, self.lookahead_generate_table):
            attached_kernel = []
            for item, las in zip(kernel, lasyms):
                attached_kernel += ((*item, s) for s in las)
            lr1_itemset_collection.append(self.closure(attached_kernel))

        self.lr1_itemset_collection = lr1_itemset_collection

    def construct_parsing_table(self):
        # Initialize table storage space.
        def create_table(x, y):
            return memoryview(bytearray(x * y * 4)).cast('L', (x, y))

        state_number = len(self.lr1_itemset_collection)
        self.parsing_table_action = create_table(state_number, len(self.terminals))
        self.parsing_table_goto = create_table(state_number, len(self.nonterminals))

        # Construct ACTION table for each state
        table_action = self.parsing_table_action

        def update_action(i, a, next_action, coming_terminal):
            prev_action = table_action[i, ~a]

            if prev_action == next_action:
                return

            if prev_action == 0:
                table_action[i, ~a] = next_action
                return

            raise RuntimeError('Parsing table error')
            # return solve_conflict(prev_action, next_action,
            #                       self.terminal_symbols[~coming_terminal])

        for i, I in enumerate(self.lr1_itemset_collection):
            for prod_idx, dot_pos, las in I:
                prod_exp = self.prods[prod_idx]

                # case 1) shift
                if dot_pos < len(prod_exp) and prod_exp[dot_pos] < 0:
                    a = prod_exp[dot_pos]
                    j = self.goto_track[i][a]
                    update_action(i, a, j << 2 | 3, a)

                # case 2) accept
                elif prod_exp[0] == 0 and dot_pos == len(prod_exp):
                    a = self.terminal_map['EOF']
                    update_action(i, a, 1, a)

                # case 3) reduce
                elif dot_pos == len(prod_exp):
                    a = las
                    update_action(i, a, prod_idx << 2 | 2, a)

                # case 4) error
                else:
                    pass  # Each cell is set with "0" by default.

        # Construct GOTO table for each state
        for i in range(len(self.lr1_itemset_collection)):
            for A, j in self.goto_track[i].items():
                if A >= 0:
                    self.parsing_table_goto[i, A] = j

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

    def stringify_lr1_item(self, item):
        prod, dot_pos, lasym = item
        prod_exp = self.prods[prod]
        lasym_str = self.terminals[~lasym] if lasym < 0 else lasym
        return self.stringify_production(prod_exp, dot_pos) + ' , ' + str(lasym_str)

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

    def print_lr1_itemset_collection(self):
        for i, itemset in enumerate(self.lr1_itemset_collection):
            print(f'C[{i}]')
            print(*('    ' + self.stringify_lr1_item(t) for t in itemset), sep='\n')

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


