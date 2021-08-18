from com import SYMBOL_HELPER_EOF, SYMBOL_HELPER_SI, SYMBOL_HELPER_ERROR
from slr import GrammarSlr


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


class ReduceToken(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return 'ReduceToken(%r, %r)' % (self.type, self.value)


class NonterminalSymbol(object):
    def __init__(self, desc):
        self.desc = desc

    def __repr__(self):
        return '<%s>' % (str(self.desc), )

    def __format__(self, *args, **kwargs):
        return self.desc.__format__(*args, **kwargs)


def is_nonterminal_name(s):
    return type(s) is str and s[0].islower()


def fetch_helper_symbol(sym_dict, desc):
    if desc in sym_dict:
        return sym_dict[desc]
    s = sym_dict[desc] = NonterminalSymbol(desc)
    return s


def _generate_prod_adder(store):
    def mark_production(*args):
        if len(args) == 0:
            raise TypeError('')

        if isinstance(args[1], list):
            rights = args[1]
        elif isinstance(args[1], tuple):
            rights = [args[1]]
        else:
            rights = [args[1:]]

        left = args[0]
        if not is_nonterminal_name(left):
            raise ValueError('The left side of a production must be an nonterminal.')
        left = fetch_helper_symbol(store.nonterminals, left)

        def collect_right_symbols(s):
            if s == 'error':
                # SYMBOL_HELPER_ERROR will be inserted later
                return SYMBOL_HELPER_ERROR
            elif is_nonterminal_name(s):
                # this is an nonterminal
                return fetch_helper_symbol(store.nonterminals, s)
            else:
                # this is a terminal
                if s not in store.terminals:
                    store.terminals.append(s)
                return s

        prods = [(left, *map(collect_right_symbols, r)) for r in rights]

        def append_production(f):
            store.hdlrs += [f] * len(prods)
            store.prods += prods
            return f

        return append_production

    return mark_production


def _normalize_precedence(user_prec):
    prec_map = {}
    for idx, term in enumerate(user_prec):
        assoc = term[0]
        level = idx + 1
        prec_map.update({s: (s, assoc, level) for s in term[1:]})
    return prec_map


class MetaHelperStore(object):
    def __init__(self):
        self.prods = []
        self.hdlrs = []
        self.nonterminals = {}
        self.terminals = []


class MetaParser(type):
    stores = {}

    @classmethod
    def __prepare__(meta, name, bases, *kwds):
        store = meta.stores[name] = MetaHelperStore()
        return {'__': _generate_prod_adder(store)}

    def __init__(cls, name, bases, namespace, **kwds):
        meta = cls.__class__
        store = meta.stores[name]
        is_base = len(bases) == 0

        cls._prods = store.prods
        cls._hdlrs = store.hdlrs

        if hasattr(cls, 'start'):
            start_symbol = fetch_helper_symbol(store.nonterminals, cls.start)
            del cls.start
        elif len(cls._prods) > 0:
            start_symbol = store.prods[0][0]
        else:
            start_symbol = None

        if start_symbol:
            cls._prods.insert(0, (SYMBOL_HELPER_SI, start_symbol))
            cls._hdlrs.insert(0, None)

        cls._nonterminals = list(store.nonterminals.values())
        cls._nonterminals.insert(0, SYMBOL_HELPER_SI)
        cls._terminals = store.terminals
        cls._terminals.insert(0, SYMBOL_HELPER_ERROR)
        cls._terminals.insert(0, SYMBOL_HELPER_EOF)

        cls._precedence_map = _normalize_precedence(getattr(cls, 'precedence', []))

        cls._error_cb = lambda msg: print(msg)

        # TODO: errf errok

        # Generate the grammar table
        if not is_base:
            grm = cls.grammar = GrammarSlr()
            grm.set_grammar(prods=cls._prods,
                            terminal_symbols=cls._terminals,
                            nonterminal_symbols=cls._nonterminals)
            grm.generate_analysis_table()


class Parser(metaclass=MetaParser):
    def __init__(self, debug=None, logger=None):
        cls = self.__class__

        self.errorok = True
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
            logger.debug('>> Terminals (%d):' % (len(cls._terminals), ))
            logger.debug('   ' + repr(cls._terminals))
            logger.debug('>> Nonterminals (%d):' % (len(cls._nonterminals), ))
            logger.debug('   ' + repr(cls._nonterminals))
            logger.debug('>> Productions (%d):' % (len(cls._prods), ))
            logger.debug('   ' + repr(cls._prods))

    def errok(self):
        self.errorok = True

    def restart(self):
        self.state_stack[:] = [0]
        self.symbol_stack[:] = [SYMBOL_HELPER_EOF]

    def parse(self, token_stream):
        cls = self.__class__
        logger = self.logger
        grmtab_action, grmtab_goto =\
            cls.grammar._table_action, cls.grammar._table_goto

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
                        look = ReduceToken(SYMBOL_HELPER_EOF, None)
                    else:
                        pass
                    if logger:
                        logger.debug('Get token (extern): ' + str(look))

            if logger:
                logger.debug('Processing token: ' + str(look))
                logger.debug('Current state: ' + str(state))
                # print(symbol_stack)
                # print(state_stack)

            symbol_idx = cls._terminals.index(look.type)
            action = grmtab_action[state, symbol_idx]
            action_type = action & 3
            action_value = action >> 2

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
                if logger:
                    logger.debug('Action::Reduce')

                # reduce symbols on the stack with a production
                prod_idx = action_value
                prod_exp = cls._prods[prod_idx]
                prod_right_length = len(prod_exp) - 1
                prod_left = prod_exp[0]
                prod_left_idx = cls._nonterminals.index(prod_left)

                tslice = [None] + [x.value for x in symbol_stack[-prod_right_length:]]
                try:
                    cls._hdlrs[prod_idx](self, tslice)
                except SyntaxError:
                    self.errorok = False
                    cls._error_cb('Syntax error.')
                    look_stack.append(look)
                    look = ReduceToken(SYMBOL_HELPER_ERROR, 'error')
                    continue

                del state_stack[-prod_right_length:]
                del symbol_stack[-prod_right_length:]

                goto_state = grmtab_goto[state_stack[-1], prod_left_idx]
                state_stack.append(goto_state)
                state = goto_state

                nonterminal_token = ReduceToken(prod_left, tslice[0])
                symbol_stack.append(nonterminal_token)

                continue

            if action_type == 1:
                if logger:
                    logger.debug('Action::Accept')

                return symbol_stack[-1].value

            if action_type == 0:
                if logger:
                    logger.debug('Action::Error')

                if look.type == SYMBOL_HELPER_ERROR:
                    # Report an exception if this error cannot be handled
                    if len(state_stack) <= 1:  # TODO
                        raise SyntaxError('Unhandled error.')
                    state_stack.pop()
                    state = state_stack[-1]
                    symbol_stack.pop()
                else:
                    if error_count == 0:
                        cls._error_cb('Syntax error.')  # TODO
                    error_count = 3
                    look_stack.append(look)
                    look = ReduceToken(SYMBOL_HELPER_ERROR, 'error')

                continue

            raise RuntimeError('Pison runtime inner error.')


