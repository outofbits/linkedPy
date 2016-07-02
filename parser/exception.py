# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>


class ParsersError(Exception):
    def __init__(self, error_msg, parser_errors=None):
        super(ParsersError, self).__init__()
        self.error_msg = error_msg
        self.parser_errors = parser_errors

    def message(self):
        return '\n'.join([p.message() for p in self.parser_errors])


class ParserError(Exception):
    """ This exception is intended to be raised, when an error . """

    def __init__(self, err_msg: str):
        super(ParserError, self).__init__()
        self.err_msg = err_msg

    def message(self):
        return '%s: %s' % (self.__class__.__name__, self.err_msg)


class IndentationError(ParserError):
    """ This exception is intended to be raised, when an indention error has been detected. """
    pass


class SyntaxError(ParserError):
    """ This exception is intended to be raised, when a syntax exception has been detected. """

    def __init__(self, err_msg: str, lexdata: str, lineno: int, lexpos: int, origin=None):
        """ Initialize this ParserException with the given parameters.

        :param err_msg: the error message of the parser.
        :param lexdata: the whole lexer data to show the line of the error.
        :param lexpos: the lexer position, where the parser exception was detected.
        :param linepos: the line position, where the parser exception was detected.
        :param origin: the origin of the program code like a file name.
        """
        super(SyntaxError, self).__init__(err_msg)
        self.origin = origin
        self.lineno = lineno
        self.linepos, self.line = SyntaxError._get_line(lexdata, lexpos)

    @staticmethod
    def _get_line(lexdata, lexpos: int) -> (int, str):
        line = ''
        c = lexdata[lexpos - 1]
        off = 1
        while c != '\n':
            line += c
            off += 1
            c = lexdata[lexpos - off]
        line = line[::-1]
        n = 0
        while lexdata[lexpos + n] != '\n':
            line += lexdata[lexpos + n]
            n += 1
        return off, line

    def message(self):
        return '%sline %d\n\t%s\n\t%s\n%s: %s' % (
            '%s in ' % self.origin if self.origin is not None else '', self.lineno, self.line,
            '%s^' % (' ' * (self.linepos - 1)), self.__class__.__name__, self.err_msg)