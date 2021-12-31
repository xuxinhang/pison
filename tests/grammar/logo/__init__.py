import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
if '.' not in sys.path:
    sys.path.insert(0, '.')
from collections import namedtuple
from plex import Lexer
from pison import Parser


class LogoLexer(Lexer):
    for s in ['>', '<', '=', '[', ']', '+', '-', '*', '/', ':', ',', '"', ';']:
        __('\\'+s)(s, lambda t: t)
    for k in ['to', 'end', 'repeat', 'if', 'make', 'print', 'fd', 'forward',
              'bk', 'backward', 'rt', 'right', 'lt', 'left', 'cs', 'clearscreen',
              'pu', 'penup', 'pd', 'pendown', 'ht', 'hideturtle', 'st', 'showturtle',
              'home', 'stop', 'label', 'setxy', 'random', 'for']:
        __(k)('_' + k, lambda t: t)
    __(r'[a-zA-Z][a-zA-Z0-9_]*')('STRING', str)
    __(r'"[a-zA-Z][a-zA-Z0-9_]*')('STRINGLITERAL', lambda _: _[1:])
    __(r'[0-9]+')('NUMBER', int)
    __(r';[^\r\n]*')(None)  # COMMENT
    __(r'\r?\n')('EOL')
    __(r'[ \t]+')(None)  # WS


LogoTreeProg = namedtuple('Prog', ['lines'])
LogoProcedureInvocation = namedtuple('ProcedureInvocation', ['name', 'args'])
LogoProcedureDeclarations = namedtuple('ProcedureDeclarations', ['name', 'args', 'lines'])
LogoTreeRepeat = namedtuple('Repeat', ['count', 'body'])
LogoTreeBlock = namedtuple('Block', ['cmds'])
LogoTreeIfe = namedtuple('Ife', ['head', 'body'])
LogoTreeComparison = namedtuple('Comparison', ['left', 'op', 'right'])
LogoTreeMake = namedtuple('Make', ['name', 'value'])
LogoTreePrint = namedtuple('Print', ['value'])
LogoTreeName = namedtuple('Name', ['name'])
LogoTreeValue = namedtuple('Value', ['type', 'value'])
LogoTreeSigExp = namedtuple('SigExp', ['op', 'rd'])
LogoTreeMulExp = namedtuple('MulExp', ['left', 'op', 'right'])
LogoTreeAddExp = namedtuple('AddExp', ['left', 'op', 'right'])
LogoTreeDeref = namedtuple('Deref', ['name'])
LogoTreeCommand = namedtuple('Command', ['name', 'arg1', 'arg2', 'arg3'],
                             defaults=[None, None, None])
LogoTreeFore = namedtuple('Fore', ['name', 'start', 'step', 'end', 'body'])
LogoTreeNumber = namedtuple('Number', ['value'])


class LogoParser(Parser):
    with __('prog') as ___:
        @___('line?_EOL+', 'line')
        def _(self, s): s[1].extend(s[2]); s[0] = LogoTreeProg(lines=s[1])
        @___('line?_EOL+')
        def _(self, s): s[0] = LogoTreeProg(lines=s[1])

    with __('line?_EOL+') as ___:
        @___('line?_EOL+', 'line', 'EOL')
        def _(self, s): s[1].append(s[2]); s[0] = s[1]
        @___('line?_EOL+', 'EOL')
        def _(self, s): s[0] = s[1]
        @___(None)
        def _(self, s): s[0] = []

    with __('line') as ___:
        @___('cmd+')
        def _(self, s): s[0] = s[1]
        @___('print')
        def _(self, s): s[0] = s[1]
        @___('procedureDeclaration')
        def _(self, s): s[0] = s[1]

    with __('cmd+') as ___:
        @___('cmd+', 'cmd')
        def _(self, s): s[1].append(s[2]); s[0] = s[1]
        @___('cmd')
        def _(self, s): s[0] = [s[1]]

    with __('cmd') as ___:
        ___('repeat')()
        ___('fd')()
        ___('bk')()
        ___('rt')()
        ___('lt')()
        ___('cs')()
        ___('pu')()
        ___('pd')()
        ___('ht')()
        ___('st')()
        ___('home')()
        ___('label')()
        ___('setxy')()
        ___('make')()
        ___('procedureInvocation')()
        ___('ife')()
        ___('stop')()
        ___('fore')()

    with __('procedureInvocation') as ___:
        @___('name', 'expression*')
        def _(self, s): s[0] = LogoProcedureInvocation(s[1], s[2])

    with __('expression*') as ___:
        @___(None)
        def _(self, s): s[0] = []
        @___('expression')
        def _(self, s): s[0] = [s[1]]
        @___('expression*', 'expression')
        def _(self, s): s[1].append(s[2]); s[0] = s[1]

    with __('procedureDeclaration') as ___:
        @___('_to', 'name', 'parameterDeclarations', 'EOL', 'line?_EOL+', '_end')
        def _(self, s):
            s[0] = LogoProcedureDeclarations(name=s[2], args=s[3], lines=s[5])
        @___('_to', 'name', 'parameterDeclarations', 'line?_EOL+', '_end')
        def _(self, s):
            s[0] = LogoProcedureDeclarations(name=s[2], args=s[3], lines=s[4])

    with __('parameterDeclarations') as ___:
        @___(None)
        def _(self, s): s[0] = []
        @___('parameterDeclarations', ':', 'name')
        def _(self, s): s[1].append(s[3]); s[0] = s[1]

    with __('func') as ___:
        ___('random')()

    with __('repeat') as ___:
        @___('_repeat', 'number', 'block')
        def _(self, s): s[0] = LogoTreeRepeat(count=s[2], body=s[3])

    with __('block') as ___:
        @___('[', 'cmd+', ']')
        def _(self, s): s[0] = LogoTreeBlock(cmds=s[2])

    with __('ife') as ___:
        @___('_if', 'comparison', 'block')
        def _(self, s): s[0] = LogoTreeIfe(head=s[2], body=s[3])

    with __('comparison') as ___:
        @___('expression', 'comparisonOperator', 'expression')
        def _(self, s): s[0] = LogoTreeComparison(left=s[1], op=s[2], right=s[3])

    with __('comparisonOperator') as ___:
        def _(self, s): s[0] = s[1]
        ___('<')(_)
        ___('>')(_)
        ___('=')(_)

    with __('make') as ___:
        @___('_make', 'STRINGLITERAL', 'value')
        def _(self, s): s[0] = LogoTreeMake(name=s[2], value=s[3])

    with __('print') as ___:
        @___('_print', 'value')
        def _(self, s): s[0] = LogoTreePrint(value=s[2])
        @___('_print', 'quotedstring')
        def _(self, s): s[0] = LogoTreePrint(value=s[2])

    # with __('quotedstring') as ___:
    #     ___('[', ']')()

    with __('name') as ___:
        @___('STRING')
        def _(self, s): s[0] = LogoTreeName(name=s[1])

    with __('value') as ___:
        @___('STRINGLITERAL')
        def _(self, s): s[0] = LogoTreeValue(type='str', value=s[1])
        @___('expression')
        def _(self, s): s[0] = LogoTreeValue(type='exp', value=s[1])
        @___('deref')
        def _(self, s): s[0] = LogoTreeValue(type='ref', value=s[1])

    with __('signExpression') as ___:
        def _(self, s): s[0] = LogoTreeSigExp(s[1], s[2])
        ___('+', 'number')(_)
        ___('+', 'deref')(_)
        ___('+', 'func')(_)
        ___('-', 'number')(_)
        ___('-', 'deref')(_)
        ___('-', 'func')(_)
        def _(self, s): s[0] = LogoTreeSigExp(None, s[1])
        ___('number')(_)
        ___('deref')(_)
        ___('func')(_)

    with __('multiplyingExpression') as ___:
        @___('signExpression')
        def _(self, s): s[0] = s[1]
        def _(self, s): s[0] = LogoTreeMulExp(s[1], s[2], s[3])
        ___('multiplyingExpression', '*', 'signExpression')(_)
        ___('multiplyingExpression', '/', 'signExpression')(_)

    with __('expression') as ___:
        @___('multiplyingExpression')
        def _(self, s): s[0] = s[1]
        def _(self, s): s[0] = LogoTreeMulExp(s[1], s[2], s[3])
        ___('multiplyingExpression', '+', 'multiplyingExpression')(_)
        ___('multiplyingExpression', '-', 'multiplyingExpression')(_)

    with __('deref') as ___:
        @___(':', 'name')
        def _(self, s): s[0] = LogoTreeDeref(name=s[2])

    def _c(self, s):
        s[0] = LogoTreeCommand(s[1],
            **{f'arg{i+1}': v for i, v in enumerate(s[2:])})

    with __('fd') as ___:
        ___('_fd', 'expression')(_c)
        ___('_forward', 'expression')(_c)

    with __('bk') as ___:
        ___('_bk', 'expression')(_c)
        ___('_backward', 'expression')(_c)

    with __('rt') as ___:
        ___('_rt', 'expression')(_c)
        ___('_right', 'expression')(_c)

    with __('lt') as ___:
        ___('_lt', 'expression')(_c)
        ___('_left', 'expression')(_c)

    with __('cs') as ___:
        ___('_cs')(_c)
        ___('_clearscreen')(_c)

    with __('pu') as ___:
        ___('_pu')(_c)
        ___('_penup')(_c)

    with __('pd') as ___:
        ___('_pd')(_c)
        ___('_pendown')(_c)

    with __('ht') as ___:
        ___('_ht')(_c)
        ___('_hideturtle')(_c)

    with __('st') as ___:
        ___('_st')(_c)
        ___('_showturtle')(_c)

    with __('home') as ___:
        ___('_home')(_c)

    with __('stop') as ___:
        ___('_stop')(_c)

    with __('label') as ___:
        ___('_label')(_c)

    with __('setxy') as ___:
        ___('_setxy', 'expression', 'expression')(_c)

    with __('random') as ___:
        ___('_random', 'expression')(_c)

    with __('fore') as ___:
        @___('_for', '[', 'name', 'expression', 'expression', 'expression', ']', 'block')
        def _(self, s):
            s[0] = LogoTreeFore(name=s[3], start=s[4], step=s[5], end=s[6], body=s[8])

    with __('number') as ___:
        @___('NUMBER')
        def _(self, s): s[0] = LogoTreeNumber(value=s[1])








