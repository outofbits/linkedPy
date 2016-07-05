# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>


class _UnknownType(object):
    """ This class represents an unknown type."""
    pass


class PrefixTable(object):
    """ This class represents a table of prefixes. """

    def __init__(self):
        self._table = dict()

    def add_prefix(self, name, iri):
        """
        Adds the given
        :param name: the name of the prefix.
        :param iri: the iri assigned to the given prefix.
        """
        self._table[name] = iri

    def prefix(self, name):
        """
        Gets the prefix with the given name or None, if there is no such prefix.
        :param name: the name of the prefix of which the iri shall be returned.
        :return: the prefix with the given name or None, if there is no such prefix.
        """
        return self._table[name]


class FunctionDescription(object):
    """ This class represents a description of a function. """

    def __init__(self, name, ast_node, total_parameters_count, fixed_parameters_count):
        """
        Initializes a function description with the given information.
        :param name: the name of the function.
        :param ast_node: the root node of the abstract syntax tree of the method.
        :param total_parameters_count: the total parameters of the function.
        :param fixed_parameters_count: the amount of fixed parameters of the function.
        """
        self.name = name
        self.ast_node = ast_node
        self.total_parameters_count = total_parameters_count
        self.fixed_parameters_count = fixed_parameters_count

    def __repr__(self):
        return '(\'Name\' : %s, \'AST\' : %s, \'Total parameters\' : %s, \'Fixed parameters\' : %s)' % (
            self.name, self.ast_node, self.total_parameters_count, self.fixed_parameters_count)


class VariableDescription(object):
    """ This class represents a description of a variable including type and value."""

    def __init__(self, name, type=None, value=None):
        """
        Initializes a description of a variable with the given name; stores the given details type and value, which are
        optional, to this variable.
        :param name: the name of the variable.
        :param type: the type of the variable
        :param value: the value of the variable.
        """
        self.name = name
        self.type = type
        self.value = value

    def change_value(self, value, type=_UnknownType):
        """
        Changes the value of this variable and the type. If no new type is given, the default type Unknown will be
        stored.
        :param value: the new value of the variable.
        :param type: the new type of the variable.
        """
        self.value = value
        self.type = type

    def __repr__(self):
        return '(\'Name\' : %s, \'Type\' : %s, \'Value : %s\')' % (self.name, self.type, self.value)


class Environment(object):
    """
    This class represents an environment that manages the function and variable table of the program or a certain
    area of it.
    """

    def __init__(self, parent_environment=None):
        """
        Initializes an environment, which may have a parent environment (e.g. local environment of function with the
        global environment as parent.)
        :param parent_environment: the parent environment of the environment that shall be initialized.
        """
        self._function_table = dict()
        self._variable_table = dict()
        self._prefix_table = dict()
        self._parent_environment = parent_environment

    def insert_function(self, name: str, ast_node, total_parameters_count: int, fixed_parameters_count: int):
        """
        Inserts a function with the given name and the root node of the abstract syntax tree of the method.
        :param name: the name of the function.
        :param ast_node: the root node of the abstract syntax tree of the method.
        :param total_parameters_count: the total amount of parameters of the method.
        :param fixed_parameters_count: the amount of fixed parameters.
        :return: the newly created function definition.
        """
        function_description = FunctionDescription(name, ast_node, total_parameters_count, fixed_parameters_count)
        self._function_table[name] = function_description
        self.insert_variable(name, value=function_description)
        return function_description

    def get_local_function(self, name: str) -> FunctionDescription:
        """
        Gets the function description of the variable with the given name, if it can be found in this environment,
        otherwise None is returned.

        :param name: the name of the function of which the function description shall be returned.
        :return: the function description of the function with the given name, or None, if it can not be found.
        """
        if name in self._function_table.keys():
            return self._function_table[name]
        return None

    def get_function(self, name: str) -> FunctionDescription:
        """
        Gets the function description of the variable with the given name, if it can be found in this environment or
        ancestors, otherwise None is returned.

        :param name: the name of the function of which the function description shall be returned.
        :return: the function description of the function with the given name, or None, if it can not be found.
        """
        if name in self._function_table.keys():
            return self._function_table[name]
        return self._parent_environment.get_function(name) if self._parent_environment is not None else None

    def insert_variable(self, name: str, type=_UnknownType, value=None):
        """
        Inserts a variable with the given name, value and type into this environment. If a variable with this name
        already exist,
        :param name: the name of the variable that shall be inserted.
        :param type: the type of the variable that shall be inserted.
        :param value: the value of the variable that shall be inserted.
        :return: the newly created variable definition.
        """
        var_description = VariableDescription(name, type, value);
        self._variable_table[name] = var_description
        return var_description

    def get_local_variable(self, name: str) -> VariableDescription:
        """
        Gets the variable description of the variable with the given name, if it can be found in this environment,
        otherwise None is returned.

        :param name: the name of the variable of which the variable description shall be returned.
        :return: the variable description of the variable with the given name, or None, if it can not be found.
        """
        if name in self._variable_table.keys():
            return self._variable_table[name]
        return None

    def get_variable(self, name: str) -> VariableDescription:
        """
        Gets the variable description of the variable with the given name, if it can be found in this environment or
        the ancestors, otherwise None is returned.

        :param name: the name of the variable of which the variable description shall be returned.
        :return: the variable description of the variable with the given name, or None, if it can not be found.
        """
        if name in self._variable_table.keys():
            return self._variable_table[name]
        return self._parent_environment.get_variable(name) if self._parent_environment is not None else None

    def insert_prefix(self, name: str, iri: str):
        """
        Inserts the prefix with the given name and corresponding iri.
        :param name: the name of the prefix.
        :param iri: the iri of the prefix with the given name.
        :return:
        """
        self._prefix_table[name] = iri
        self.insert_variable(name, _UnknownType, iri)

    def get_prefix(self, name: str) -> str:
        """
        Gets the iri of the prefix with the given name.
        :param name: the name of the prefix.
        :return: the iri of the prefix with the given name or None, if it can not be found.
        """
        return self._prefix_table[name] if name in self._prefix_table else None

    def __repr__(self):
        return 'Environment {Variable-Table: %s,  Function-Table: %s}' % (self._variable_table, self._function_table)


class ProgramContainer(object):
    """ This class contains the program data with additional information like the origin. """

    def __init__(self, program_string: str, origin: str = None):
        self.origin = origin
        self.program_string = program_string
        self._p_container = dict()
        for line_no, line in enumerate(program_string.splitlines()):
            self._p_container[line_no + 1] = line

    def __getitem__(self, item) -> str:
        if isinstance(item, slice):
            start = item.start if item.start is not None else 1
            stop = item.stop if item.stop is not None else len(self._p_container)
            return [self._p_container[x] for x in range(start, stop,
                                                        item.step if item.step is not None else 1)]
        elif isinstance(item, int):
            if item not in self._p_container:
                raise IndexError('Index %s is out of bounds.' % item)
            return self._p_container[item]
        else:
            raise TypeError('The given type %s is inappropriate.' % item.__class__.__name__)


class ProgramPeephole(object):
    """ This class represents a peephole that points at a certain area of the program. """

    def __init__(self, program_container: ProgramContainer, start_line_no: int, end_line_no: int):
        self.program_container = program_container
        self.start_line_no = start_line_no
        self.end_line_no = end_line_no

    def program_snippet(self):
        """
        Returns the program code between the start and end line.
        :return: the program code between the start and end line.
        """
        if self.start_line_no == self.end_line_no:
            return self.program_container[self.start_line_no]
        else:
            return '\n'.join(self.program_container[self.start_line_no:self.end_line_no])

    def __repr__(self):
        return '{%sLine: (%d,%d)}' % (
            'Origin: %s ' % self.program_container.origin if self.program_container.origin is not None else '',
            self.start_line_no, self.end_line_no)


class ProgramStack(object):
    """ This class represents the course of the program execution. """

    def __init__(self, program_stack, peephole: ProgramPeephole):
        self.prev = program_stack
        self.peephole = peephole

    def get_stack(self, max_len=10) -> [ProgramPeephole]:
        """
        Goes back in the given program stack and returns all entries as a list, where the most recent entry is at the
        beginning.
        :param max_len: the maximal number of entries that shall be returned.
        :return: the traceback of the program stack as list of peepholes that point at certain areas of the program.
        """
        stack = [self.peephole]
        stack_prev = self.prev
        while stack_prev is not None and max_len > 0:
            stack.append(stack_prev.peephole)
            stack_prev = stack_prev.prev
            max_len -= 1
        return stack