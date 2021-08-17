from plex import Lexer
from pison import Parser


class MyLexer(Lexer):
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


class MyParser(Parser):
    @__('calclist', ('exp', 'EOL'))
    def t_calclist_1(self, p):
        print('>> ' + str(p[1]))

    @__('calclist', ('calclist', 'exp', 'EOL'))
    def t_calclist_2(self, p):
        print('>> ' + str(p[2]))

    @__('exp', 'factor')
    def t_exp_factor(self, p):
        p[0] = p[1]

    @__('exp', 'exp', 'ADD', 'factor')
    def t_exp_add(self, p):
        p[0] = p[1] + p[3]

    @__('exp', 'exp', 'SUB', 'factor')
    def t_exp_sub(self, p):
        p[0] = p[1] - p[3]

    @__('factor', 'term')
    def t_factor_term(self, p):
        p[0] = p[1]

    @__('factor', 'factor', 'MUL', 'term')
    def t_factor_mul(self, p):
        p[0] = p[1] * p[3]

    @__('factor', 'factor', 'DIV', 'term')
    def t_factor_div(self, p):
        p[0] = p[1] / p[3]

    @__('term', 'NUMBER')
    def t_term_number(self, p):
        p[0] = p[1]


lex = MyLexer()
par = MyParser()
print(par._hdlrs)
print(par._prods)
print(par._nonterminals)
# print(par._precedence_map)
par.grammar.print_analysis_table()

lex.input('''\
2+3
24/6+9
8*8*8*8*8*8*8*8
''')

par.parse(lex)

