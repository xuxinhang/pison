import unittest
from plex import Lexer
from pison import Parser


class TestcaseVariousGrammar(unittest.TestCase):
    def test_the_recursive_production_set_including_an_empty_production(self):
        class TestLexer(Lexer):
            __(r'\(')('(', str)
            __(r'\)')(')', str)
            __(r'\[')('[', str)
            __(r'\]')(']', str)
            __(r'\<')('<', str)
            __(r'\>')('>', str)
            __(r'\{')('{', str)
            __(r'\}')('}', str)

        class TestParser(Parser):
            def _c(self, s): s[0] = s[1] + s[2] + s[3]
            def _n(self, s): s[0] = ''

            with __('pair') as ___:
                ___(None)(_n)
                ___('(', 'pair', ')')(_c)
                ___('[', 'pair', ']')(_c)
                ___('<', 'pair', '>')(_c)
                ___('{', 'pair', '}')(_c)

        lex = TestLexer()
        par = TestParser()
        testcases = ['<[([([<()>])])]>', '(((())))']
        results = []
        for c in testcases:
            lex.input(c)
            results.append(par.parse(lex))
        self.assertEqual(testcases, results)

