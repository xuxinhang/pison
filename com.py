# common used

class AugmentedSymbol:
    _augmented = True

    def __init__(self, desc):
        self.desc = desc

    def __repr__(self):
        return '#' + str(self.desc) + '#'

    def __format__(self, *args, **kwargs):
        return self.desc.__format__(*args, **kwargs)


AUG_SYMBOL_EOF = AugmentedSymbol('$end')
AUG_SYMBOL_SI = AugmentedSymbol('S\'')
AUG_SYMBOL_ERROR = AugmentedSymbol('error')
