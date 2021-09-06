import unittest
from plex import Lexer
from pison import Parser


def __():
    pass


class StorageLogger():
    def __init__(self):
        self.store = []

    def debug(self, msg):
        self.store.append(msg)

    def info(self, msg):
        self.store.append(msg)

    def warning(self, msg):
        self.store.append(msg)

    def error(self, msg):
        self.store.append(msg)

    critical = debug


class CalcLexer(Lexer):
    __(r'\+')('PLUS')
    __(r'\-')('MINUS')
    __(r'\*')('TIMES')
    __(r'\/')('DIVIDE')
    __(r'\=')('EQUALS')
    __(r'\(')('LPAREN')
    __(r'\)')('RPAREN')
    __(r'[a-zA-Z_][a-zA-Z0-9_]*')('NAME')

    @__(r'\d+')
    def t_NUMBER(self, t):
        t.type = 'NUMBER'
        try:
            t.value = int(t.value)
        except ValueError:
            print("Integer value too large %s" % t.value)
            t.value = 0
        return t

    __([' ', r'\t'])(None)

    @__(r'\n+')
    def t_newline(self, t):
        t.lexer.lineno += t.value.count("\n")

    @__('__error__')
    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)


def skipNotImplementedTest():
    return unittest.skip('Not Implemented')


def ignoreTest(f):
    return None


def deprecatedTest(f):
    return None


class TestcaseErrorWarning(unittest.TestCase):
    @skipNotImplementedTest()
    def test_badargs(self):
        pass

    @deprecatedTest
    def test_badid(self):
        pass

    def test_badprec(self):
        try:
            class TestParser(Parser):
                precedence = 42

                @__('statement', 'NAME', 'EQUALS')
                def p_statement_assign(t):
                    pass

            TestParser()
        except Exception as e:
            self.assertEqual(e.args[0], 'precedence field must be an iterable.')
        else:
            self.fail()

    def test_badprec2(self):
        try:
            class TestParser(Parser):
                precedence = [42, ('left', 'TIMES', 'DIVIDE'), ('right', 'UMINUS')]

                @__('statement', 'NAME', 'EQUALS')
                def p_statement_assign(t):
                    pass
            TestParser()
        except Exception as e:
            self.assertEqual(e.args[0], 'precedence item must have one element at least.')
        else:
            self.fail()

    def test_badprec3(self):
        try:
            class TestParser(Parser):
                precedence = [('left', 'TIMES', 'DIVIDE', 'UMINUS'), ('right', 'UMINUS')]

                @__('statement', 'NAME', 'EQUALS')
                def p_statement_assign(t):
                    pass
            TestParser()
        except Exception as e:
            self.assertEqual(e.args[0],
                             'The terminal \'UMINUS\' has already been assigned with precedence.')
        else:
            self.fail()

    def test_badrule(self):
        try:
            class TestParser(Parser):
                @__()
                def p_statement_assign(t):
                    pass
            TestParser()
        except Exception as e:
            self.assertEqual(e.args[0], 'The left side of the production must be provided.')
        else:
            self.fail()

    @skipNotImplementedTest()
    def test_badtok(self):
        pass

    @deprecatedTest
    def test_dup(self):
        pass

    @skipNotImplementedTest()
    def test_error1(self):
        try:
            class TestParser(Parser):
                @__('statement', 'NAME', 'EQUALS')
                def p_statement_assign(t):
                    pass

                def error(self, msg):
                    pass
        except Exception as e:
            self.assertEqual(e.args[0], '')
        else:
            self.fail()

    @skipNotImplementedTest()
    def test_error2(self):
        try:
            class TestParser(Parser):
                @__('statement', 'NAME', 'EQUALS')
                def p_statement_assign(t):
                    pass

                def error():
                    pass
        except Exception as e:
            self.assertEqual(e.args[0], '')
        else:
            self.fail()

    def test_error3(self):
        try:
            class TestParser(Parser):
                @__('statement', 'NAME', 'EQUALS')
                def p_statement_assign(t):
                    pass

                error = 'bla'
        except Exception as e:
            self.assertEqual(e.args[0], 'Error handler must be callable.')
        else:
            self.fail()

    def test_error4(self):
        try:
            class TestParser(Parser):
                @__('error', 'NAME')
                def p_error_handler(t):
                    pass
        except ValueError as e:
            self.assertEqual(e.args[0],
                             'The left side of the production can\'t be the terminal #error#')
        else:
            self.fail()

    def test_error5(self):
        tracker = []

        def print(t):
            tracker.append(t)

        class TestParser(Parser):
            precedence = [('left', 'PLUS', 'MINUS'),
                          ('left', 'TIMES', 'DIVIDE'),
                          ('right', 'UMINUS')]

            @__('statement', 'expression')
            def p_statement_expr(self, t):
                print(t[1])

            @__('expression', [('expression', 'PLUS', 'expression'),
                               ('expression', 'MINUS', 'expression'),
                               ('expression', 'TIMES', 'expression'),
                               ('expression', 'DIVIDE', 'expression')])
            def p_expression_binop(self, t):
                if   t[2] == '+': t[0] = t[1] + t[3]  # noqa
                elif t[2] == '-': t[0] = t[1] - t[3]  # noqa
                elif t[2] == '*': t[0] = t[1] * t[3]  # noqa
                elif t[2] == '/': t[0] = t[1] / t[3]  # noqa

            @__('expression', 'MINUS', 'expression', '%prec', 'UMINUS')
            def p_expression_uminus(self, t):
                t[0] = -t[2]

            @__('expression', 'LPAREN', 'expression', 'RPAREN')
            def p_expression_group(self, t):
                t[0] = t[2]

            @__('expression', 'LPAREN', 'error', 'RPAREN')
            def p_expression_group_error(self, t):
                t[0] = 0

            @__('expression', 'NUMBER')
            def p_expression_number(self, t):
                t[0] = t[1]

            def error(self, msg):
                print(msg)

        logger = StorageLogger()
        lex = CalcLexer()
        lex.lineno = 1
        par = TestParser(logger=logger)

        lex.input('(4*5) + (9 8 7) + - 6 + 7')
        par.parse(lex)

        self.assertEqual('\n'.join(map(str, tracker)), 'syntax error\n21')

    def test_error6(self):
        tracker = []

        def tracker_print(x):
            return tracker.append(str(x))

        class TestParser(Parser):
            # Parsing rules
            precedence = [('left', 'PLUS', 'MINUS'),
                          ('left', 'TIMES', 'DIVIDE'),
                          ('right', 'UMINUS')]

            @__('statements', 'statements', 'statement')
            def p_statements(self, t):
                pass

            @__('statements', 'statement')
            def p_statements_1(self, t):
                pass

            @__('statement', 'LPAREN', 'expression', 'RPAREN')
            def p_statement_expr(self, t):
                tracker_print(t[2])

            @__('statement', 'LPAREN', 'error', 'RPAREN')
            def p_statement_expr_error(self, t):
                pass

            @__('expression', [('expression', 'PLUS', 'expression'),
                               ('expression', 'MINUS', 'expression'),
                               ('expression', 'TIMES', 'expression'),
                               ('expression', 'DIVIDE', 'expression')])
            def p_expression_binop(self, t):
                if   t[2] == '+': t[0] = t[1] + t[3]  # noqa
                elif t[2] == '-': t[0] = t[1] - t[3]  # noqa
                elif t[2] == '*': t[0] = t[1] * t[3]  # noqa
                elif t[2] == '/': t[0] = t[1] / t[3]  # noqa

            @__('expression', 'MINUS', 'expression', '%prec', 'UMINUS')
            def p_expression_uminus(self, t):
                t[0] = -t[2]

            @__('expression', 'NUMBER')
            def p_expression_number(self, t):
                t[0] = t[1]

            def error(self, p):  # TODO
                tracker_print(p)
                # tracker.append("Line %d: Syntax error at '%s'" % (p.lineno, p.value))

        lex = CalcLexer()
        par = TestParser()
        lex.input('(3 + 4)\n(4 + * 5 - 6 + *)\n(10 + 11)')
        par.parse(lex)
        self.assertEqual('\n'.join(tracker), '7\nsyntax error\n21')

    def test_error7(self):
        tracker = []

        def print(t):
            tracker.append(t)

        class TestParser(Parser):
            precedence = [('left','PLUS','MINUS'),
                          ('left','TIMES','DIVIDE'),
                          ('right','UMINUS')]

            @__('statements', 'statements', 'statement')
            def p_statements(self, t):
                pass

            @__('statements', 'statement')
            def p_statements_1(self, t):
                pass

            @__('statement', 'LPAREN', 'NAME', 'EQUALS', 'expression', 'RPAREN')
            def p_statement_assign(self, p):
                print("%s=%s" % (p[2],p[4]))

            @__('statement', 'LPAREN', 'expression', 'RPAREN')
            def p_statement_expr(self, t):
                print(t[1])

            @__('expression', [('expression', 'PLUS', 'expression'),
                               ('expression', 'MINUS', 'expression'),
                               ('expression', 'TIMES', 'expression'),
                               ('expression', 'DIVIDE', 'expression')])
            def p_expression_binop(self, t):
                if t[2] == '+'  : t[0] = t[1] + t[3]
                elif t[2] == '-': t[0] = t[1] - t[3]
                elif t[2] == '*': t[0] = t[1] * t[3]
                elif t[2] == '/': t[0] = t[1] / t[3]

            @__('expression', 'MINUS', 'expression', '%prec', 'UMINUS')
            def p_expression_uminus(self, t):
                t[0] = -t[2]

            @__('expression', 'NUMBER')
            def p_expression_number(self, t):
                t[0] = t[1]

            @__('statement', 'error', 'RPAREN')
            def p_error(self, p):
                print("Line %d: Syntax error at '%s'" % (0, 0))

            def error(self, p):
                pass

        logger = StorageLogger()
        lex = CalcLexer()
        lex.lineno = 1
        par = TestParser(logger=logger)

        lex.input('(a = 3 + 4)\n(b = 4 + * 5 - 6 + *)\n(c = 10 + 11)')
        par.parse(lex)

        self.assertEqual('\n'.join(map(str, tracker)),
                         "a=7\nLine 0: Syntax error at '0'\nc=21")

    @skipNotImplementedTest()
    def test_inf(self):
        pass

    @deprecatedTest
    def test_literal(self):
        pass

    @deprecatedTest
    def test_misplaced(self):
        pass

    @skipNotImplementedTest()
    def test_missing1(self):
        pass

    @deprecatedTest
    def test_nested(self):
        pass

    @deprecatedTest
    def test_nodoc(self):
        pass

    @deprecatedTest
    def test_noerror(self):
        pass

    @deprecatedTest
    def test_nop(self):
        pass

    @deprecatedTest
    def test_notfunc(self):
        pass

    @deprecatedTest
    def test_notok(self):
        pass

    @skipNotImplementedTest()
    def test_rr(self):
        pass

    @skipNotImplementedTest()
    def test_rr_unused(self):
        pass

    def test_simple(self):
        class TestParser(Parser):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.names = {}

            # Parsing rules
            precedence = [('left','PLUS','MINUS'),
                        ('left','TIMES','DIVIDE'),
                        ('right','UMINUS')]

            @__('statement', 'NAME', 'EQUALS', 'expression')
            def p_statement_assign(self, t):
                self.names[t[1]] = t[3]

            @__('statement', 'expression')
            def p_statement_expr(self, t):
                print(t[1])

            @__('expression', [('expression', 'PLUS', 'expression'),
                                ('expression', 'MINUS', 'expression'),
                                ('expression', 'TIMES', 'expression'),
                                ('expression', 'DIVIDE', 'expression')])
            def p_expression_binop(t):
                if t[2] == '+'  : t[0] = t[1] + t[3]
                elif t[2] == '-': t[0] = t[1] - t[3]
                elif t[2] == '*': t[0] = t[1] * t[3]
                elif t[2] == '/': t[0] = t[1] / t[3]

            @__('expression', 'MINUS', 'expression', '%prec', 'UMINUS')
            def p_expression_uminus(self, t):
                t[0] = -t[2]

            @__('expression', 'LPAREN', 'expression', 'RPAREN')
            def p_expression_group(self, t):
                t[0] = t[2]

            @__('expression', 'NUMBER')
            def p_expression_number(self, t):
                t[0] = t[1]

            @__('expression', 'NAME')
            def p_expression_name(self, t):
                try:
                    t[0] = self.names[t[1]]
                except LookupError:
                    print("Undefined name '%s'" % t[1])
                    t[0] = 0

            def error(self):
                print("Syntax error at '%s'" % ('?',))

        self.assertTrue(True)

    @skipNotImplementedTest()
    def test_sr(self):
        pass

    @skipNotImplementedTest()
    def test_term1(self):
        pass

    @deprecatedTest
    def test_unicode_literals(self):
        pass

    @skipNotImplementedTest()
    def test_unused(self):
        pass

    @skipNotImplementedTest()
    def test_unused_rule(self):
        pass

    def test_uprec(self):
        try:
            class TestParser(Parser):
                @__('expression', 'MINUS', 'expression', '%prec', 'UMINUS')
                def p_expression_uminus(t):
                    t[0] = -t[2]
        except ValueError as e:
            self.assertEqual(e.args[0], '%prec symbol is not assigned with precedence.')
        else:
            self.fail()

    def test_uprec2(self):
        try:
            class TestParser(Parser):
                @__('expression', 'MINUS', 'expression', '%prec')
                def p_expression_uminus(t):
                    t[0] = -t[2]
        except ValueError as e:
            self.assertEqual(e.args[0], 'Invalid %prec parameter.')
        else:
            self.fail()

    @skipNotImplementedTest()
    def test_prec1(self):
        pass

    def test_param1(self):
        try:
            class TestParser(Parser):
                @__('expression', 'MINUS', 'expression', '%qrec')
                def p_expression_uminus(t):
                    t[0] = -t[2]
        except ValueError as e:
            self.assertEqual(e.args[0], 'Unexpected %')
        else:
            self.fail()

