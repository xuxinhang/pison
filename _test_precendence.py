from plex import Lexer
from pison import Parser


class CalculatorLexer(Lexer):
    __(r'\+')('ADD')
    __(r'\-')('SUB')
    __(r'\*')('MUL')
    __(r'\/')('DIV')

    @__(r'[0-9]+')
    def t_sub(self, t):
        t.type = 'NUMBER'
        t.value = int(t.value)
        return t

    __(r'\n')('EOL')
    __(r'[ \t]')(None)


class CalculatorParser(Parser):
    precedence = [
        ('left', 'ADD', 'SUB'),
        ('left', 'MUL', 'DIV'),
        ('nonassoc', 'UMINUS'),
    ]

    @__('calclist', 'EOL')
    def t_calclist_0(self, p):
        print('>>')

    @__('calclist', ('exp', 'EOL'))
    def t_calclist_1(self, p):
        print('>> ' + str(p[1]))

    @__('calclist', ('calclist', 'exp', 'EOL'))
    def t_calclist_2(self, p):
        print('>> ' + str(p[2]))

    @__('exp', 'NUMBER')
    def t_exp_number(self, p):
        p[0] = p[1]

    @__('exp', 'exp', 'ADD', 'exp')
    def t_exp_add(self, p):
        p[0] = p[1] + p[3]

    @__('exp', 'exp', 'SUB', 'exp')
    def t_exp_sub(self, p):
        p[0] = p[1] - p[3]

    @__('exp', 'exp', 'MUL', 'exp')
    def t_exp_mul(self, p):
        p[0] = p[1] * p[3]

    @__('exp', 'exp', 'DIV', 'exp')
    def t_exp_div(self, p):
        if p[3] == 0:
            raise SyntaxError('DIV BY ZERO')
        p[0] = p[1] / p[3]

    @__('exp', 'SUB', 'exp', '%prec', 'UMINUS')
    def t_uminus(self, p):
        p[0] = -p[2]

    @__('calclist', [('error', 'EOL'), ('calclist', 'error', 'EOL')])
    def t_error(self, p):
        pass

    def error(self, msg):
        print(f'[SYNTAX ERROR] {msg}')


lex = CalculatorLexer()
par = CalculatorParser(debug=True)
par.grammar.print_analysis_table()

lex.input('''
24/6+9
8*+
10/0+1*9
12/-6*3
-8*-8/-8/-8
''')

par.parse(lex)

