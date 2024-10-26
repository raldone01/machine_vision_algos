import unittest.mock as mock


# https://github.com/testing-cabal/mock/issues/139#issuecomment-122128815
# This Mock class is picklable.
class BetterMock(mock.Mock):
    def __reduce__(self):
        return (mock.Mock, ())
