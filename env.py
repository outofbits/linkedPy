# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

import re
import hashlib

from exception import TypeError as ITypeError
from linkedtypes import resource, triple, graph

FILE_NAME_ENDING_REGEX = re.compile(r'(\..*)*$', re.UNICODE)


class _UnknownType(object):
    """ This class represents an unknown type."""
    pass


class Function(object):
    """ This class represents a description of a function. """

    def __init__(self, name, ast_node, environment, total_parameters, default_parameters, doc: str = None):
        """
        Initializes a function description with the given information.
        :param name: the name of the function.
        :param ast_node: the root node of the abstract syntax tree of the function.
        :param environment: the enclosed environment of the function.
        :param total_parameters: the list of all parameters.
        :param default_parameters: the list of parameters with default expressions.
        :param doc: the optional documentation of the function.
        """
        self.name = name
        self.ast_node = ast_node
        self.environment = environment
        self.total_parameters = total_parameters
        self.default_parameters = default_parameters
        self.doc = doc

    @property
    def total_parameters_count(self):
        return len(self.total_parameters)

    @property
    def optional_parameters_count(self):
        return len(self.default_parameters)

    @property
    def mandatory_parameters_count(self):
        return self.total_parameters_count - self.optional_parameters_count

    def __name__(self):
        """
        Gets the name of the function.
        :return: the name of the function.
        """
        return self.name

    def __defaults__(self):
        """
        A tuple containing default argument values for those arguments that have defaults, or None if no arguments have
        a default value.
        :return: tuple containing default argument values for those arguments that have defaults, or None if no
        arguments have a default value
        """
        if self.default_parameters is not None:
            return (default.execute(self.environment, None) for default in self.default_parameters.values())
        else:
            return None

    def __module__(self):
        """
        Gets the enclosing module name.
        :return: the enclosing module name.
        """
        pass

    def __code__(self):
        """
        Gets the code of this function in form of an abstract syntax tree.
        :return: the code of the function in form of an abstract syntax tree.
        """
        return self.ast_node

    def __doc__(self):
        """
        Gets the function’s documentation string, or None if unavailable; not inherited by subclasses
        :return: the function’s documentation string, or None if unavailable; not inherited by subclasses
        """
        return self.doc

    def __call__(self, program_stack, *positional_args, **keyword_args):
        local_environment = Environment(self.environment)
        # Check that the arguments are given properly.
        total_len = len(positional_args) + len(keyword_args)
        argument_repr = ('argument', 'arguments')
        if total_len > self.total_parameters_count or total_len < self.mandatory_parameters_count:
            raise ITypeError('%d %s given, but \'%s\' takes %d %s%s.' % (
                total_len, argument_repr[total_len > 1], self.name, self.total_parameters_count,
                argument_repr[self.total_parameters_count > 1],
                ', where %d %s are optional' % (self.optional_parameters_count, argument_repr[
                    self.optional_parameters_count > 1]) if self.optional_parameters_count > 0 else ''),
                             program_stack)
        # Handle positional arguments
        parameters_names_to_activate = set(self.total_parameters.values())
        for index, positional_arg_value in enumerate(positional_args):
            positional_arg_name = self.total_parameters[index]
            local_environment.insert_variable(self.total_parameters[index], _UnknownType, positional_arg_value)
            parameters_names_to_activate.remove(positional_arg_name)
        # Handle the given keyword arguments.
        for keyword_arg_name in keyword_args:
            if keyword_arg_name not in parameters_names_to_activate:
                raise ITypeError(
                    'Keyword argument \'%s\' is not applicable for this call of \'%s\'.' % (
                        keyword_arg_name, self.name),
                    program_stack)
            local_environment.insert_variable(keyword_arg_name, _UnknownType, keyword_args[keyword_arg_name])
            parameters_names_to_activate.remove(keyword_arg_name)
        # Handle the parameters with default expressions that were not be given.
        for default_arg_name in parameters_names_to_activate:
            if default_arg_name not in self.default_parameters:
                raise ITypeError(
                    'Fixed argument \'%s\' was not given for this call of \'%s\'.' % (default_arg_name, self.name),
                    program_stack)
            local_environment.insert_variable(default_arg_name, _UnknownType,
                                              self.default_parameters[default_arg_name].execute(local_environment,
                                                                                                program_stack).value)
        return self.__code__().execute(local_environment, program_stack).value


class Variable(object):
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

    def __set_value__(self, value, type=_UnknownType):
        """
        Sets the value of the variable as well as the type of the variable. The type is optional and _UnknownType is the
        default.
        :param value: the value that the variable should have.
        :param type: the type that the value of the variable has.
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
        self._variable_table = dict()
        self._prefix_table = dict()
        self._parent_environment = parent_environment

    def insert_variable(self, name: str, type=_UnknownType, value=None):
        """
        Inserts a variable with the given name, value and type into this environment. If a variable with this name
        already exist,
        :param name: the name of the variable that shall be inserted.
        :param type: the type of the variable that shall be inserted.
        :param value: the value of the variable that shall be inserted.
        :return: the newly created variable definition.
        """
        self._variable_table[name] = Variable(name, type, value)

    def get_local_variable(self, name: str) -> Variable:
        """
        Gets the variable description of the variable with the given name, if it can be found in this environment,
        otherwise None is returned.

        :param name: the name of the variable of which the variable description shall be returned.
        :return: the variable description of the variable with the given name, or None, if it can not be found.
        """
        if name in self._variable_table.keys():
            return self._variable_table[name]
        return None

    def get_variable(self, name: str) -> Variable:
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
        return 'Environment {Variable-Table: %s}' % self._variable_table


class GlobalEnvironment(Environment):
    internal_functions = {
        'print': print,
        'len': len,
        # Collections
        'list': list,
        'tuple': tuple,
        'resource': resource,
        'triple': triple,
        'graph': graph,
    }

    internal_attributes = {
        '__name__': None,
        '__file__': None,

    }

    def __init__(self, name, file_path=None):
        super(GlobalEnvironment, self).__init__()
        self.internal_attributes['__name__'] = name
        self.internal_attributes['__file__'] = file_path
        self._setup()

    def _setup(self):
        for i_function in self.internal_functions:
            self.insert_variable(name=i_function, type=Function, value=self.internal_functions[i_function])
        for i_attribute in self.internal_attributes:
            self.insert_variable(name=i_attribute, type=Variable, value=self.internal_attributes[i_attribute])


class ProgramContainer(object):
    """ This class contains the program data with additional information like the origin. """

    def __init__(self, program_string: str, origin: str = None, program_dir: str = None,
                 program_basename: str = None):
        """
        Initializes the program container.
        :param program_string: the string of the program.
        :param origin: the path to the program file that contains the given program string.
        :param program_dir: the path to the directory containing the program file that contains the given program string.
        :param program_basename: the name of the program file that contains the given program string.
        """
        self.program_string = program_string
        self.origin = origin
        self.program_dir = program_dir
        self.program_basename = program_basename
        self.program_name = FILE_NAME_ENDING_REGEX.sub('', program_basename) if self.program_basename else None
        self._hash = None
        self._p_container = dict()
        self._p_line_counter = 0
        for line_no, line in enumerate(program_string.splitlines()):
            self._p_container[line_no + 1] = line
            self._p_line_counter += 1

    @property
    def hash_digest(self) -> bytes:
        if self._hash is None:
            self._hash = bytes.fromhex(hashlib.sha224(str.encode(self.program_string, 'utf-8')).hexdigest())
        return self._hash

    def __setitem__(self, key, value):
        raise ValueError('The %s is immutable.' % self.__class__.__name__)

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

    def __len__(self):
        return self._p_line_counter

    def __hash__(self):
        return hash(self.hash_digest)


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

    def get_stack(self, max_len: int = 10) -> [ProgramPeephole]:
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
