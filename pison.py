from com import AUG_SYMBOL_EOF, AUG_SYMBOL_SI, AUG_SYMBOL_ERROR


class DefaultLogger(object):
    def __init__(self, *, f):
        if f is None:
            from sys import stderr
            self.f = stderr
        else:
            self.f = f

    def debug(self, msg):
        self.f.write('DEBUG: ' + msg + '\n')

    def info(self, msg):
        self.f.write('INFO : ' + msg + '\n')

    def warning(self, msg):
        self.f.write('WARN : ' + msg + '\n')

    def error(self, msg, *args, **kwargs):
        self.f.write('ERROR: ' + msg + '\n')

    critical = debug


# Class Production
class Production(object):
    """
    This class keeps anything about production, including
    left part, right part and prec part, etc.
    """
    def __init__(self, left, right, *, prec=None, hdlr=None):
        self._tuple = (left, *right)
        self.prec = prec

    def __getitem__(self, idx):
        return self._tuple[idx]

    def __len__(self):
        return len(self._tuple)

    def __repr__(self):
        return repr(self._tuple[0]) + ' -> '\
            + ' '.join(map(repr, self._tuple[1:]))\
            + (('(%prec ' + repr(self.prec) + ')') if self.prec is not None else '')


class ReduceToken(object):
    """
    Inner used Token class.
    """
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return 'ReduceToken(%r, %r)' % (self.type, self.value)


def _default_rule_action(self, p):
    """ When the action is empty, use it. """
    p[0] = p[1]


class ProductionAdder(object):
    @staticmethod
    def presetStore(store):
        return lambda *args: ProductionAdder(store, *args)

    def __init__(self, store, *args):
        self.store = store

        if len(args) == 0:
            raise ValueError('The left side of the production must be provided.')

        def _norm_symbol(s):
            if s == 'error':
                return AUG_SYMBOL_ERROR
            return s

        left = _norm_symbol(args[0])

        if len(args) == 1:
            rests = [()]  # one empty production
        elif type(args[1]) is list:
            rests = [r if type(r) is tuple else (r,) for r in args[1]]
        elif type(args[1]) is tuple:
            rests = [args[1]]
        else:
            rests = [args[1:]]

        def _parse_production_right(rest):
            # fields of production right
            ret_right = None
            ret_prec = None

            # mark is something like "%name", we record their name and position.
            last_mark_name = None
            last_mark_pos = -1

            # enumerate through items then split and save each field into variables.
            for i in range(len(rest) + 1):
                r = None if i == len(rest) else rest[i]
                if (r is None or r == '') or\
                    (type(r) is str and len(r) > 1 and r[0] == '%'):
                    if last_mark_name is None:
                        ret_right = tuple(map(_norm_symbol, rest[last_mark_pos+1:i]))
                    elif last_mark_name == 'prec':
                        if last_mark_pos == i-2:
                            ret_prec = rest[last_mark_pos+1]
                        else:
                            raise ValueError('Invalid %prec parameter.')
                    else:
                        raise ValueError('Unexpected %')
                    last_mark_name = r and r[1:]
                    last_mark_pos = i
                if r is None or r == '':
                    break

            return Production(left, ret_right, prec=ret_prec)

        prods = list(map(_parse_production_right, rests))

        # save value for later calling
        self.prod_left = left
        self.prods = prods

    def __call__(self, f=None):
        if f is None:
            f = _default_rule_action
        elif not callable(f):
            raise TypeError('The production handler must be callable.')
        prods = self.prods
        self.store.prods += prods
        self.store.hdlrs += [f] * len(prods)
        return f

    def __enter__(self):
        # only production left is usable.
        return lambda *args:\
            self.__class__(self.store, self.prod_left, *args)

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _default_error_cb(self, msg):
    print(msg)  # TODO


class AbsProductionTuple(tuple):
    pass


class MetaHelperStore(object):
    """
    This class provides a temporary storage space.
    """
    def __init__(self):
        self.prods = []
        self.hdlrs = []
        self.nonterminals = {}
        self.terminals = []


class MetaParser(type):
    """
    Metaclass here.
    """
    stores = {}

    @classmethod
    def __prepare__(meta, name, bases, *kwds):
        # prepare the namespace to provide a production adder usable in class body
        # create independent store for different classes.
        store = meta.stores[name] = MetaHelperStore()
        return {'__': ProductionAdder.presetStore(store)}

    def __init__(cls, name, bases, namespace, **kwds):
        meta = cls.__class__
        store = meta.stores[name]
        if len(bases) == 0:
            return  # do not compile the base class.

        # Set productions
        cls._prods = store.prods

        # Set production handlers
        cls._hdlrs = store.hdlrs

        # Validate and collect grammar symbols in productions
        cls._nonterminals = [AUG_SYMBOL_SI]
        cls._terminals = [AUG_SYMBOL_EOF, AUG_SYMBOL_ERROR]

        for p in cls._prods:
            if p[0] in cls._terminals:
                raise ValueError('The left side of the production can\'t be the terminal '
                                 + repr(p[0]))
            if p[0] not in cls._nonterminals:
                cls._nonterminals.append(p[0])

        for p in cls._prods:
            for t in p[1:]:
                if t not in cls._nonterminals and t not in cls._terminals:
                    cls._terminals.append(t)

        # Generate precedence map
        cls_precedence = getattr(cls, 'precedence', [])
        try:
            iter(cls_precedence)
        except Exception:
            raise TypeError('precedence field must be an iterable.')
        cls._precedence_map = prec_map = {}
        for idx, item in enumerate(cls_precedence):
            level = idx + 1
            try:
                assoc = item[0]
            except Exception:
                raise TypeError('precedence item must have one element at least.')
            if assoc not in {'left', 'right', 'nonassoc'}:
                raise ValueError('Invalid precedence assoc value.')
            for s in item[1:]:
                if s in prec_map:
                    raise ValueError('The terminal %r has already been assigned with precedence.'
                                     % (s,))
                prec_map[s] = (s, assoc, level)

        # Set start symbol
        if hasattr(cls, 'start'):
            if cls.start is None or cls.start not in cls._nonterminals:
                raise ValueError('The start symbol is not a valid nonterminal.')
            start_symbol = cls.start
            del cls.start
        elif len(cls._prods) > 0:
            start_symbol = cls._prods[0][0]
        else:
            raise ValueError('Must assign one production at least.')
        cls._prods.insert(0, Production(AUG_SYMBOL_SI, (start_symbol,)))
        cls._hdlrs.insert(0, None)

        # Validate %prec field
        #   ensure every %prec symbol is attached with precedence.
        for p in cls._prods:
            if p.prec is not None and p.prec not in cls._precedence_map:
                raise ValueError('%prec symbol is not assigned with precedence.')

        # Abstract grammar productions
        #   encode production items with numbers.
        def get_symbol_idx(s):
            try:
                return cls._nonterminals.index(s)
            except Exception:
                return ~cls._terminals.index(s)

        cls._abs_prods = []
        for p in cls._prods:
            a = AbsProductionTuple(map(get_symbol_idx, p))
            a.prec = p.prec  # HACK
            cls._abs_prods.append(a)

        # Generate the grammar table
        cls.grammar_engine = getattr(cls, 'grammar_engine', 'lalr')
        if cls.grammar_engine == 'slr':
            from slr import GrammarSlr
            grm = GrammarSlr()
            grm.set_grammar(prods=cls._prods,
                            terminal_symbols=cls._terminals,
                            nonterminal_symbols=cls._nonterminals,
                            precedence_map=cls._precedence_map,
                            abs_prods=cls._abs_prods),
            grm.compile()
        elif cls.grammar_engine == 'lalr':
            from lalr import GrammarLalr
            grm = GrammarLalr()
            grm.set_grammar(productions=cls._abs_prods,
                            terminals=cls._terminals,
                            nonterminals=cls._nonterminals,
                            precedence_map=cls._precedence_map)
            grm.compile()
        else:
            raise ValueError('An unknown grammar engine')
        cls.grammar = grm

        # Register error handler routine
        if hasattr(cls, 'error'):
            if not callable(cls.error):
                raise TypeError('Error handler must be callable.')
            cls._error_cb = cls.error
            del cls.error
        else:
            cls._error_cb = _default_error_cb


class Parser(metaclass=MetaParser):
    def __init__(self, debug=None, logger=None):
        cls = self.__class__

        self._errok_mark = False
        self.state_stack = []
        self.symbol_stack = []

        # configurate the logger
        self.logger = None
        if logger:
            self.logger = logger
        else:
            if debug is True:
                from sys import stderr
                self.logger = DefaultLogger(f=stderr)
            elif debug is False or debug is None:
                self.logger = None
            else:
                self.logger = DefaultLogger(f=debug)

        # debug logger
        logger = self.logger
        if logger:
            logger.debug('Parser Grammar Information --->')
            logger.debug('Terminals (%d):' % (len(cls._terminals), ))
            logger.debug('. ' + repr(cls._terminals))
            logger.debug('Nonterminals (%d):' % (len(cls._nonterminals), ))
            logger.debug('. ' + repr(cls._nonterminals))
            logger.debug('Productions (%d):' % (len(cls._prods), ))
            for p in cls._prods:
                logger.debug('. ' + repr(p))

    def errok(self):
        self._errok_mark = True

    def restart(self):
        self.state_stack[:] = [0]
        self.symbol_stack[:] = [AUG_SYMBOL_EOF]

    def parse(self, token_stream):
        cls = self.__class__
        logger = self.logger

        if hasattr(cls.grammar, '_table_action'):  # TODO
            grmtab_action = cls.grammar._table_action
        elif hasattr(cls.grammar, 'parsing_table_action'):
            grmtab_action = cls.grammar.parsing_table_action

        if hasattr(cls.grammar, '_table_goto'):
            grmtab_goto = cls.grammar._table_goto
        elif hasattr(cls.grammar, 'parsing_table_goto'):
            grmtab_goto = cls.grammar.parsing_table_goto

        state_stack = self.state_stack = [0]
        symbol_stack = self.symbol_stack = []
        state = 0

        look_stack = []
        look = None
        error_count = 0

        while True:
            if look is None:
                if look_stack:
                    look = look_stack.pop()
                    if logger:
                        logger.debug('Get token (look_stack): ' + str(look))
                else:
                    try:
                        look = next(token_stream)
                    except StopIteration:
                        look = ReduceToken(AUG_SYMBOL_EOF, None)
                    if logger:
                        logger.debug('Get token (extern): ' + str(look))

            try:
                symbol_idx = cls._terminals.index(look.type)
            except Exception:
                raise SyntaxError('Unknown token with type ' + str(look.type))
            action = grmtab_action[state, symbol_idx]
            action_type, action_value = (action & 3), (action >> 2)

            # mark a syntax error raised manually by the reducer funtion
            reduce_manual_error = None

            if logger:
                logger.debug('Current state: %s >> Coming token: %s' % (state, look))

            if action_type == 3:
                if logger:
                    logger.debug('Action::Shift')

                # shift a symbol into the stack
                state = action_value
                state_stack.append(state)
                symbol_stack.append(look)
                look = None

                if error_count:
                    error_count -= 1
                continue

            if action_type == 2:
                # reduce symbols on the stack with a production
                prod_idx = action_value
                prod_exp = cls._abs_prods[prod_idx]
                prod_right_length = len(prod_exp) - 1
                prod_left_idx = prod_exp[0]
                prod_left_sym = cls._nonterminals[prod_left_idx]

                if logger:
                    logger.debug('Action::Reduce (%d) %r' % (prod_idx, cls._prods[prod_idx]))

                tslice = [None]
                if prod_right_length > 0:
                    tslice.extend(x.value for x in symbol_stack[-prod_right_length:])
                try:
                    cls._hdlrs[prod_idx](self, tslice)
                except SyntaxError as e:
                    # mark a syntax error raised manually
                    reduce_manual_error = e
                else:
                    if prod_right_length > 0:
                        del state_stack[-prod_right_length:]
                        del symbol_stack[-prod_right_length:]

                    goto_state = grmtab_goto[state_stack[-1], prod_left_idx]
                    state_stack.append(goto_state)
                    state = goto_state

                    nonterminal_token = ReduceToken(prod_left_sym, tslice[0])
                    symbol_stack.append(nonterminal_token)

                    continue

            if action_type == 0 or reduce_manual_error:
                if logger:
                    if reduce_manual_error:
                        logger.debug('To handle syntax error raised manually.')
                    else:
                        logger.debug('Action::Error')

                if error_count == 0 or self._errok_mark:
                    self._errok_mark = False
                    err_msg = reduce_manual_error.args[0]\
                        if reduce_manual_error else 'syntax error'
                    cls._error_cb(self, err_msg)
                error_count = 3

                # TODO: two special case

                if look.type == AUG_SYMBOL_ERROR:
                    # Report an exception if this error cannot be handled
                    if len(state_stack) <= 1:  # TODO
                        if hasattr(look, '_except'):
                            raise SyntaxError(*look._except.args)
                        else:
                            raise SyntaxError('Unrecoverable syntax error.')
                    state_stack.pop()
                    state = state_stack[-1]
                    symbol_stack.pop()
                else:
                    if len(symbol_stack) >= 1 and symbol_stack[-1].type == AUG_SYMBOL_ERROR:
                        # Just discard this token if the top of symbol stack has already been an error.
                        look = None
                    else:
                        # Replace the lookahead token with the newly-created error symbol.
                        look_stack.append(look)
                        look = ReduceToken(AUG_SYMBOL_ERROR, look)
                        look._except = reduce_manual_error

                reduce_manual_error = None  # reset
                continue

            if action_type == 1:
                if logger:
                    logger.debug('Action::Accept')
                return getattr(symbol_stack[-1], 'value', None)

            raise RuntimeError('Pison runtime inner error.')


