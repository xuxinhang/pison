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
    """This class keeps anything about production, including
    left part, right part and prec part, etc. """
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
    """Inner used Token class."""

    # we mainly use member names from Flex/Bison, ...
    def __init__(self, char, lval, lloc):
        self.char = char
        self.lval = lval
        self.lloc = lloc

    # ... but still hold compatible member names from PLY
    def type_getter(self): return self.char  # noqa
    def type_setter(self, x): self.char = x  # noqa
    type = property(type_getter, type_setter)
    def value_getter(self): return self.lval  # noqa
    def value_setter(self, x): self.lval = x  # noqa
    value = property(value_getter, value_setter)

    def __repr__(self):
        return 'ReduceToken(%r, %r)' % (self.char, self.lval)


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
    """This class provides a temporary storage space."""
    def __init__(self):
        self.prods = []
        self.hdlrs = []
        self.nonterminals = {}
        self.terminals = []


class MetaParser(type):
    """Metaclass here."""
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
        else:
            cls.error = cls._error_cb = _default_error_cb


class Parser(metaclass=MetaParser):
    def __init__(self, debug=None, logger=None):
        cls = self.__class__

        self._error_call_flag = False
        self.nerrs = 0
        self.recovering_status = 0
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

    # def error(self, *args, **kwargs):
    #     return self.__class__._error_cb(*args, **kwargs)

    def errok(self):
        """Immediately quit recovering mode."""
        self.recovering_status = 0

    @property
    def recovering(self):
        """Whether parser is recovering from a syntax error."""
        return bool(self.recovering_status)

    def restart(self):
        self.state_stack[:] = [0]
        self.symbol_stack[:] = [AUG_SYMBOL_EOF]

    def ERROR(self):
        """Raise error explicitly."""
        self._error_call_flag = True

    def parse(self, token_stream):
        cls = self.__class__
        logger = self.logger

        # prepare grammar tables
        if hasattr(cls.grammar, '_table_action'):  # TODO
            grmtab_action = cls.grammar._table_action
        elif hasattr(cls.grammar, 'parsing_table_action'):
            grmtab_action = cls.grammar.parsing_table_action
        else:
            raise ValueError('Fail to load grammar paring table ACTION')
        if hasattr(cls.grammar, '_table_goto'):
            grmtab_goto = cls.grammar._table_goto
        elif hasattr(cls.grammar, 'parsing_table_goto'):
            grmtab_goto = cls.grammar.parsing_table_goto
        else:
            raise ValueError('Fail to load grammar paring table GOTO')

        # reset runtime variables
        self.nerrs = 0
        state_stack = self.state_stack
        symbol_stack = self.symbol_stack
        state_top = 0
        state_stack[:] = [state_top]
        symbol_stack[:] = []
        las_stash = []
        las_next = None
        manual_error = False

        while True:
            if las_next is None:
                if las_stash:
                    las_next = las_stash.pop()
                    if logger:
                        logger.debug('Get stashed token: ' + str(las_next))
                else:
                    try:
                        las_next = next(token_stream)
                    except StopIteration:
                        las_next = ReduceToken(AUG_SYMBOL_EOF, None, None)
                    if logger:
                        logger.debug('Fetch new token: ' + str(las_next))

            try:
                las_next_idx = cls._terminals.index(las_next.char)
            except Exception:
                raise SyntaxError('Unknown token type ' + str(las_next.char))
            action = grmtab_action[state_top, las_next_idx]
            action_type = action & 3
            action_value = action >> 2

            if logger:
                logger.debug('State = %s << Lookahead = %s' % (state_top, las_next))

            if action_type == 1:
                if logger:
                    logger.debug('Action::Accept')
                return getattr(symbol_stack[-1], 'value', None)

            if action_type == 3:
                if logger:
                    logger.debug('Action::Shift')
                # shift a symbol into the stack
                state_top = action_value
                state_stack.append(state_top)
                symbol_stack.append(las_next)
                las_next = None
                if self.recovering_status > 0:
                    self.recovering_status -= 1
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

                lval_slice = [None]
                lloc_slice = [None]
                if prod_right_length > 0:
                    lval_slice.extend(x.lval for x in symbol_stack[-prod_right_length:])
                    lloc_slice.extend(x.lloc for x in symbol_stack[-prod_right_length:])
                    del state_stack[-prod_right_length:]
                    del symbol_stack[-prod_right_length:]
                hdlrs = cls._hdlrs[prod_idx]
                if hdlrs.__code__.co_argcount == 3:
                    hdlrs(self, lval_slice, lloc_slice)
                elif hdlrs.__code__.co_argcount == 2:
                    hdlrs(self, lval_slice)
                elif hdlrs.__code__.co_argcount == 1:
                    hdlrs(self)

                if self._error_call_flag:
                    self._error_call_flag = False
                    manual_error = True
                    state_top = state_stack[-1]
                    # continue with the next branch case
                else:
                    goto_state = grmtab_goto[state_stack[-1], prod_left_idx]
                    state_stack.append(goto_state)
                    state_top = goto_state
                    symbol_stack.append(
                        ReduceToken(prod_left_sym, lval_slice[0], lloc_slice[0]))
                    continue

            if action_type == 0 or manual_error:
                err_msg = 'syntax error'
                if logger:
                    logger.debug('Action::Error')

                if manual_error:
                    manual_error = False
                else:
                    # report this error if not in recovering mode
                    if self.recovering_status == 0:
                        self.nerrs += 1
                        cls._error_cb(self, err_msg)

                self.recovering_status = 3

                if las_next.char != AUG_SYMBOL_ERROR:
                    if len(symbol_stack) and symbol_stack[-1].char == AUG_SYMBOL_ERROR:
                        # Drop this token if the toppest symbol already an error.
                        las_next = None
                    else:
                        las_stash.append(las_next)
                        las_next = ReduceToken(AUG_SYMBOL_ERROR, err_msg, None)
                else:
                    if len(state_stack) <= 1:
                        raise SyntaxError(err_msg)
                    state_stack.pop()
                    state_top = state_stack[-1]
                    symbol_stack.pop()
                continue

            # unreachable here
