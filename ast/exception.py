class ExecutionError(Exception):
    """ This class represents an exception that indicates an error during the execution of the abstract syntax tree."""

    def __init__(self, line: str, lineno: int, linepos: int, err_msg: str = '', origin=None):
        """ Initialize this ExecutionError with the given parameters.

        :param line: the line, where the parser exception was detected.
        :param lineno: the line number, where the parser exception was detected.
        :param linepos: the line position, where the parser exception was detected.
        :param origin: the origin of the program code like a file name.
        """
        super(ExecutionError, self).__init__()
        self.origin = origin
        self.line = line
        self.lineno = lineno
        self.linepos = linepos
        self.err_msg = err_msg

    def message(self):
        return '%sline %d\n\t%s\n\t%s\n%s: %s' % (
            '%s in ' % self.origin if self.origin is not None else '', self.lineno, self.line,
            '%s^' % (' ' * (self.linepos - 1)), self.__class__.__name__, self.err_msg)


class VariableError(ExecutionError):
    def __init__(self, line: str, lineno: int, linepos: int, err_msg: str = '', origin=None):
        super(VariableError, self).__init__(line, lineno, linepos, err_msg, origin)
