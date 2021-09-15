from com import AUG_SYMBOL_EOF


class GrammarBase():
    pass


class GrammarLalr(GrammarBase):
    def __init__(self):
        # grammar definition
        self.prods = []
        self.terminals = []
        self.nonterminals = []
        # grammar table
        self.itemset_collection_lr0 = []
        self._table_goto = None
        self._table_action = None
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

        # patch grammar with something used internally
        self.terminals.append('#')
        self.terminal_map['propagate_placeholder'] = ~(len(self.terminals) - 1)

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

    def _follows(self):
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
            for prod_exp in self.prods:
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

    def closure_lr0(self, I):  # noqa: E741
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

    def goto_lr0(self, I, X, *, kernel=False):  # noqa: E741
        cache_key = (id(I), X)
        # if cache_key in self._goto_cache:
        #     return self._goto_cache[cache_key]

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
        C, D = [self.closure_lr0([(0, 1)])], [[(0, 1)]]
        v, w = 0, len(C)
        while v < w:
            I = C[v]  # noqa: E741
            # we care no symbols but following symbols.
            following_symbols = []
            for prod_idx, dot_pos in I:
                prod_exp = self.prods[prod_idx]
                if dot_pos < len(prod_exp):
                    if (s := prod_exp[dot_pos]) not in following_symbols:
                        following_symbols.append(s)
            for X in following_symbols:
                k, g = self.goto_lr0(I, X, kernel=True)
                if len(g) > 0 and g not in C:
                    C.append(g)
                    D.append(k)
                    w += 1
            v += 1

        self.itemset_collection_lr0 = C
        self.itemset_kernel_collection_lr0 = D

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

    # -----------------
    # Construct LALR itemset from LR(0) itemset
    # -----------------
    def attach_lookahead(self):
        itemset_transition_track = self.itemset_transition_track

        kernel_collection = self.itemset_kernel_collection_lr0
        lookahead_generate_table = [[[] for _ in K] for K in kernel_collection]
        lookahead_propagate_table = [[[] for _ in K] for K in kernel_collection]

        symbol_propagate_placeholder = self.terminal_map['propagate_placeholder']

        for K_idx, K in enumerate(kernel_collection):
            for ki, (k_prod, k_dot_pos) in enumerate(K):
                J = self.closure([(k_prod, k_dot_pos, symbol_propagate_placeholder)])
                print('closure')
                print(*map(lambda i: self.stringify_item_lr1(i), J), sep='\n')
                for prod, dot_pos, a in J:
                    prod_exp = self.prods[prod]
                    if dot_pos >= len(prod_exp):
                        continue
                    X = prod_exp[dot_pos]
                    goto_kernel_idx = itemset_transition_track[K_idx][X]
                    target_idx = kernel_collection[goto_kernel_idx].index((prod, dot_pos+1))
                    if a == symbol_propagate_placeholder:
                        lookahead_propagate_table[K_idx][ki].append((goto_kernel_idx, target_idx))
                    else:
                        lookahead_generate_table[goto_kernel_idx][target_idx].append(a)

        for _ in lookahead_generate_table[0]:
            _.append(self.terminal_map['EOF'])

        self.lookahead_generate_table = lookahead_generate_table
        self.lookahead_propagate_table = lookahead_propagate_table

    def propagate_lookahead(self):
        lookahead_generate_table = self.lookahead_generate_table
        lookahead_propagate_table = self.lookahead_propagate_table

        cnt = 1
        while cnt:
            cnt = 0
            for K, propagate_table_K in enumerate(lookahead_propagate_table):
                for item, propagate_table_K_item in enumerate(propagate_table_K):
                    for target_kernel, target_item in propagate_table_K_item:
                        for s in lookahead_generate_table[K][item]:
                            if s not in lookahead_generate_table[target_kernel][target_item]:
                                lookahead_generate_table[target_kernel][target_item].append(s)
                                cnt += 1

    def items(self):
        K = [(0, 1)]
        I = self.closure_lr0(K)
        C, D, state_transition_tracker = [I], [K], [{}]

        v, w = 0, len(C)
        while v < w:
            I = C[v]
            # we care no symbols but following symbols.
            following_symbols = []
            for prod_idx, dot_pos in I:
                prod_exp = self.prods[prod_idx]
                if dot_pos < len(prod_exp):
                    if (s := prod_exp[dot_pos]) not in following_symbols:
                        following_symbols.append(s)
            for X in following_symbols:
                K, g = self.goto_lr0(I, X, kernel=True)
                if len(g) == 0:
                    continue
                try:
                    gi = C.index(g)
                except Exception:
                    C.append(g)
                    D.append(K)
                    state_transition_tracker[v][X] = len(C) - 1
                    state_transition_tracker.append({})
                    w += 1
                else:
                    state_transition_tracker[v][X] = gi
            v += 1

        self.itemset_collection_lr0 = C
        self.itemset_kernel_collection_lr0 = D
        self.itemset_transition_track = state_transition_tracker

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

    def stringify_item_lr0(self, item):
        prod, dot_pos = item
        prod_exp = self.prods[prod]
        return self.stringify_production(prod_exp, dot_pos)

    def stringify_item_lr1(self, item):
        prod, dot_pos, lasym = item
        prod_exp = self.prods[prod]
        lasym_str = self.terminals[~lasym] if lasym < 0 else lasym
        return self.stringify_production(prod_exp, dot_pos) + ' , ' + str(lasym_str)

    def print_itemset_collection_lr0(self):
        for i, itemset in enumerate(self.itemset_collection_lr0):
            print(f'C[{i}]')
            print(*('    ' + self.stringify_item_lr0(t) for t in itemset), sep='\n')

    def print_itemset_kernel_collection_lr0(self):
        for i, itemset in enumerate(self.itemset_kernel_collection_lr0):
            print(f'K[{i}]')
            print(*('    ' + self.stringify_item_lr0(t) for t in itemset), sep='\n')

    def print_lookahead_propagate_table(self):
        table = self.lookahead_propagate_table
        kernel_collection = self.itemset_kernel_collection_lr0
        for K_i, K in enumerate(table):
            for ki, propagate_targets in enumerate(K):
                if len(propagate_targets) == 0:
                    continue
                source_item = kernel_collection[K_i][ki]
                print('I%d:  %s' % (K_i, self.stringify_item_lr0(source_item)))
                for target in propagate_targets:
                    target_item = kernel_collection[target[0]][target[1]]
                    print('        I%d:  %s' % (target[0], self.stringify_item_lr0(target_item)))

    def print_lookahead_generate_table(self):
        table = self.lookahead_generate_table
        kernel_collection = self.itemset_kernel_collection_lr0
        for K, table_K in enumerate(table):
            for item, table_K_item in enumerate(table_K):
                print('I%d: %s : %s' % (K,
                                        self.stringify_item_lr0(kernel_collection[K][item]),
                                        ' / '.join(str(self.terminals[~s]) for s in table_K_item)))


