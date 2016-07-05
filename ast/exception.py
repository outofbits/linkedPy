class ExecutionError(Exception):
    """ This class represents an exception that indicates an error during the execution of the abstract syntax tree."""

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
                program_traceback = '\t%s\n\t\t%s\n%s' % (current_trace_line, peephole.program_snippet(), program_traceback)
        return 'Traceback (most recent call last):\n%s%s: %s' % (program_traceback, self.__class__.__name__, self.error_message)


class VariableError(ExecutionError):
    """ This class represents a error that will be thrown, if a variable cannot be accessed. """

    def __init__(self, error_message: str, program_stack, *args, **kwargs):
        super(VariableError, self).__init__(error_message, program_stack, *args, **kwargs)


class TypeError(ExecutionError):
    """ This class represents a error that will be thrown if a operation cannot be carried out with the current type."""

    def __init__(self, error_message: str, program_stack, *args, **kwargs):
        super(TypeError, self).__init__(error_message, program_stack, *args, **kwargs)
