# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>
import abc


class ParserError(Exception):
    """ This exception is intended to be raised, when an error . """

    def __init__(self, line: str, lineno: int, linepos: int, err_msg:str = ''):
        """ Initialize this ParserException with the given parameters.

        :param line: the line, where the parser exception was detected.
        :param lineno: the line number, where the parser exception was detected.
        :param linepos: the line position, where the parser exception was detected.
        """
        self.line = line
        self.lineno = lineno
        self.linepos = linepos
        self.err_msg = err_msg
        super().__init__()

    def message(self):
        return '__file__ in line %d\n%s\n%s: %s' % (self.lineno, self.line.split("\n")[0], self.__class__.__name__,
                                                    self.err_msg)


class IndentError(ParserError):
    """ This exception is intended to be raised, when an indention error has been detected. """
    pass


class SyntaxError(ParserError):
    """ This exception is intended to be raised, when a syntax exception has been detected. """
    pass
