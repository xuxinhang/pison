import sys
import unittest

if '..' not in sys.path:
    sys.path.insert(0, '..')

if '.' not in sys.path:
    sys.path.insert(0, '.')

from test_error_warning import TestcaseErrorWarning  # noqa: F401
from test_various_grammar import TestcaseVariousGrammar  # noqa: F401
unittest.main()

