
class AugmentedSymbol:
    def __init__(self, desc):
        self.desc = desc

    def __repr__(self):
        return '#' + str(self.desc) + '#'

    def __format__(self, *args, **kwargs):
        return self.desc.__format__(*args, **kwargs)


SYMBOL_HELPER_EOF = AugmentedSymbol('$end')
SYMBOL_HELPER_SI = AugmentedSymbol('S\'')
SYMBOL_HELPER_ERROR = AugmentedSymbol('error')
