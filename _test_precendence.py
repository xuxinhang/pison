from enum import Enum
from plex import Lexer
from pison import Parser


class CToken(Enum):
    POW = '^'
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    LPAR = '('
    RPAR = ')'
    UMINUS = 0
    NUMBER = 256
    EOL = 257


class CalculatorLexer(Lexer):
    __(r'\^')(CToken.POW)
    __(r'\+')(CToken.ADD)
    __(r'\-')(CToken.SUB)
    __(r'\*')(CToken.MUL)
    __(r'\/')(CToken.DIV)
    __(r'\(')(CToken.LPAR)
    __(r'\)')(CToken.RPAR)

    @__(r'[0-9]+')
    def t_sub(self, t):
        t.type = CToken.NUMBER
        t.value = int(t.value)
        return t

    __(r'\n')(CToken.EOL)
    __(r'[ \t]')(None)


class CalculatorParser(Parser):
    precedence = [
        ('left', CToken.ADD, CToken.SUB),
        ('left', CToken.MUL, CToken.DIV),
        ('nonassoc', 'UMINUS'),
        ('right', CToken.POW),
    ]

    @__('calclist', None)
    def t_calclist_0(self, p):
        print('>> EMPTY')

    @__('calclist', ('calclist', CToken.EOL))
    def t_calclist_1(self, p):
        print('>> BLANK LINE')

    @__('calclist', ('calclist', 'exp', CToken.EOL))
    def t_calclist_2(self, p):
        print('>> ' + str(p[2]))

    with __('exp') as _:
        @_(CToken.NUMBER)
        def t_exp_number(self, p):
            p[0] = p[1]

        @_('exp', CToken.ADD, 'exp')
        def t_exp_add(self, p):
            p[0] = p[1] + p[3]

        @_('exp', CToken.SUB, 'exp')
        def t_exp_sub(self, p):
            p[0] = p[1] - p[3]

        @_('exp', CToken.MUL, 'exp')
        def t_exp_mul(self, p):
            p[0] = p[1] * p[3]

        @_('exp', CToken.DIV, 'exp')
        def t_exp_div(self, p):
            if p[3] == 0:
                raise SyntaxError('DIV BY ZERO')
            p[0] = p[1] / p[3]

        @_(CToken.SUB, 'exp', '%prec', 'UMINUS')
        def t_uminus(self, p):
            p[0] = -p[2]

        @_(CToken.LPAR, 'exp', CToken.RPAR)
        def t_par_exp(self, p):
            p[0] = p[2]

        @_('exp', CToken.POW, 'exp')
        def t_pow(self, p):
            p[0] = p[1] ** p[3]

    @__('calclist', [('error', CToken.EOL),
                     ('calclist', 'error', CToken.EOL)])
    def t_error(self, p):
        pass

    def error(self, msg):
        print(f'[SYNTAX ERROR] {msg}')


lex = CalculatorLexer()
par = CalculatorParser(debug=False)
par.grammar.print_analysis_table(
    terminal_formatter=lambda t: t.value if type(t.value) is str else t.name)

lex.input('''
24/6+9
8*+
10/0+1*9
10/(0+1)*9
12/-6*3
-8*-8/-8/-8
-2^(-3)^2
''')

par.parse(lex)

