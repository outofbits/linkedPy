class ExecutionError(Exception):
    def __init__(self, error_message: str, program_stack, *args, **kwargs):
        """ Initialize this ExecutionError with the given parameters.

        :param error_message: the error message
        :param program_stack: the stack representing the course of the program execution.
        :param origin: the origin of the program code like a file name.
        """
        super(ExecutionError, self).__init__(error_message, *args, **kwargs)
        self.program_stack = program_stack.get_stack(max_len=6)
        self.error_message = error_message

    def message(self):
        program_traceback = ''
        if self.program_stack:
            # Print out the traceback of the program execution
            for peephole in self.program_stack:
                if peephole is None: continue
                current_trace_line = ''
                if peephole.start_line_no == peephole.end_line_no:
                    current_trace_line = '%sline %d' % (
                        '%s in ' % peephole.program_container.origin if peephole.program_container.origin is not None else '',
                        peephole.start_line_no)
                else:
                    current_trace_line = '%sbetween line %d and %d' % (
                        peephole.program_container.origin if peephole.program_container.origin is not None else '',
                        peephole.start_line_no, peephole.end_line_no)
                program_traceback = '\t%s\n\t\t%s\n%s' % (
                    current_trace_line, peephole.program_snippet(), program_traceback)
        return 'Traceback (most recent call last):\n%s%s: %s' % (
            program_traceback, self.__class__.__name__, self.error_message)


class VariableError(ExecutionError):
    def __init__(self, error_message: str, program_stack, *args, **kwargs):
        super(VariableError, self).__init__(error_message, program_stack, *args, **kwargs)


class TypeError(ExecutionError):
    def __init__(self, error_message: str, program_stack, *args, **kwargs):
        super(TypeError, self).__init__(error_message, program_stack, *args, **kwargs)


class InternalError(ExecutionError):
    def __init__(self, exception: Exception, program_stack):
        super(InternalError, self).__init__(str(exception), program_stack)


class PrefixError(ExecutionError):
    def __init__(self, error_message: str, program_stack, *args, **kwargs):
        super(PrefixError, self).__init__(error_message, program_stack, *args, **kwargs)


class ParserErrors(Exception):
    def __init__(self, error_msg, parser_errors=None):
        super(ParserErrors, self).__init__()
        self.error_msg = error_msg
        self.parser_errors = parser_errors

    def message(self):
        return '\n'.join([p.message() for p in self.parser_errors])


class ParserError(Exception):
    def __init__(self, *args, **kwargs):
        super(ParserError, self).__init__(*args, **kwargs)


class EOFParserError(ParserError):
    def __init__(self, err_msg: str, *args, **kwargs):
        super(EOFParserError, self).__init__(*args, **kwargs)
        self.err_msg = err_msg

    def message(self):
        return '%s: %s' % (self.__class__.__name__, self.err_msg)


class IndentationError(ParserError):
    pass


class SyntaxError(ParserError):
    def __init__(self, err_msg: str, lexdata: str, lineno: int, lexpos: int, origin: str = None, *args, **kwargs):
        """ Initialize this ParserException with the given parameters.

        :param err_msg: the error message of the parser.
        :param lexdata: the whole lexer data to show the line of the error.
        :param lexpos: the lexer position, where the parser exception was detected.
        :param linepos: the line position, where the parser exception was detected.
        :param origin: the origin of the program code like a file name.
        """
        super(SyntaxError, self).__init__(*args, **kwargs)
        self.origin = origin
        self.err_msg = err_msg
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


class IntermediateCodeError(Exception):
    def __init__(self, error_message: str, *args, **kwargs):
        super(IntermediateCodeError, self).__init__(error_message, *args, **kwargs)
        self.error_message = error_message

    def message(self):
        return self.error_message


class IntermediateCodeTransformationError(IntermediateCodeError):
    def __init__(self, error_message: str, *args, **kwargs):
        super(IntermediateCodeTransformationError, self).__init__(error_message, *args, **kwargs)


class IntermediateCodeCorruptedError(IntermediateCodeError):
    def __init__(self, error_message: str, *args, **kwargs):
        super(IntermediateCodeCorruptedError, self).__init__(error_message, *args, **kwargs)


class IntermediateCodeOutdatedError(IntermediateCodeError):
    def __init__(self, error_message: str, *args, **kwargs):
        super(IntermediateCodeOutdatedError, self).__init__(error_message, *args, **kwargs)


class IntermediateCodeConstantNotFound(IntermediateCodeError):
    def __init__(self, error_message: str, *args, **kwargs):
        super(IntermediateCodeConstantNotFound, self).__init__(error_message, *args, **kwargs)


class IntermediateCodeFileNotFound(IntermediateCodeError):
    def __init__(self, error_message: str, *args, **kwargs):
        super(IntermediateCodeFileNotFound, self).__init__(error_message, *args, **kwargs)


class IntermediateCodeLimitationError(object):
    def __init__(self, error_message: str, *args, **kwargs):
        super(IntermediateCodeLimitationError, self).__init__(error_message, *args, **kwargs)