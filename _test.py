from enum import Enum
from slr import GrammarSlr


class Ns(Enum):
    EE = 1
    E = 2
    T = 3
    F = 4


class Ts(Enum):
    PLUS = 3
    STAR = 4
    LP = 5
    RP = 6
    ID = 7


PRESET_PRODUCTIONS = [
    (Ns.EE, Ns.E),
    (Ns.E, Ns.E, Ts.PLUS, Ns.T), (Ns.E, Ns.T),
    (Ns.T, Ns.T, Ts.STAR, Ns.F), (Ns.T, Ns.F),
    (Ns.F, Ts.LP, Ns.E, Ts.RP), (Ns.F, Ts.ID),
]

grm = GrammarSlr()
grm.set_grammar(prods=PRESET_PRODUCTIONS,
                terminal_symbols=Ts, nonterminal_symbols=Ns)

print('== CLOSURE ==')
print(grm.stringify_item_collection(grm.closure([(0, 1)])))

print('== CLOSURE ==')
print(grm.stringify_item_collection(grm.closure([(3, 3)])))

print('== GOTO ==')
print(grm.stringify_item_collection(grm.goto([(0, 2), (1, 2)], Ts.PLUS)))

print('== ITEMS ==')
for i, s in enumerate(grm.items()):
    print(f'[{i}]')
    print(grm.stringify_item_collection(s))

print('== FIRST ==')
for s in grm.nonterminal_symbols:
    print(f'FIRST({s})', grm._first(s))
# for s in grm.terminal_symbols:
#     print(f'FIRST({s})', grm._first(s))

print('== FOLLOW ==')
for k, f in grm._compute_follows().items():
    print(f'FOLLOW({k})', f)

print('== LR Table ==')
grm.generate_analysis_table()
grm.print_analysis_table()

