# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

from enum import Enum
from abc import abstractmethod
from collections import namedtuple, deque
from env import Environment, Function, Variable, ProgramPeephole, ProgramStack
from exception import (VariableError, InternalError, TypeError as ITypeError, IntermediateCodeCorruptedError)
from linkedtypes import resource, triple, graph

byte_ast_dispatch = dict()


class ASTExecutionResultType(Enum):
    void_ = 1
    value_ = 2
    return_ = 3
    continue_ = 4
    break_ = 5


class ASTPrepareType(Enum):
    var_found_ = 1
    var_not_found_ = 2


ASTExecutionResult = namedtuple('ASTExecutionResult', ['type', 'value'])
ASTPrepareResult = namedtuple('ASTPrepareResult', ['type', 'value'])


class ASTNode(object):
    identifier_length = 1
    byte_separator = bytes(identifier_length)
    # peephole
    no_peephole_bytes = bytes([0x06])
    single_line_peephole = bytes([0x07])
    peephole_bytes_length = len(single_line_peephole)

    def __init__(self, peephole: ProgramPeephole = None):
        self.children = []
        self.peephole = peephole

    @abstractmethod
    def execute(self, environment: Environment, program_stack: ProgramStack) -> ASTExecutionResult:
        """
        Executes this ASTNode and returns the corresponding result of the execution.
        :param environment: the environment in which context this ASTNode shall be executed.
        :param program_stack: the program stack that represents the course of the program execution.
        :return: the result of the execution.
        """
        raise NotImplementedError('Execute-Method of %s is not implemented !' % self.__class__.__name__)

    def cache_peephole(self, constant_pool) -> bytearray:
        """
        Caches the peephole of the current abstract syntax tree node to enable convenient error messages for
        executions based on trees restored from cached intermediate code.
        :return: the byte code for the peephole.
        """
        if self.peephole is not None:
            if self.peephole.start_line_no == self.peephole.end_line_no:
                peephole_bytes = bytearray(self.single_line_peephole)
                peephole_bytes.extend(constant_pool.add(NumberNode(self.peephole.start_line_no)))
                return peephole_bytes
            else:
                peephole_bytes = bytearray(constant_pool.add(NumberNode(self.peephole.start_line_no)))
                peephole_bytes.extend(constant_pool.add(NumberNode(self.peephole.end_line_no)))
                return peephole_bytes
        return bytearray(self.no_peephole_bytes)

    @classmethod
    def construct_peephole_from_cache(cls, fd, constant_pool, program_container) -> ProgramPeephole:
        """
        Restores the peephole from the byte code the file descriptor points to and the given program container,
        containing at least the program string.
        :param fd: the file descriptor that points to the byte code representing a peephole.
        :param constant_pool: the constant pool that stores all the constants.
        :param program_container: the program container that contains at least the program string.
        :return: the peephole that has been restored from the byte code, or None if no peephole data given.
        """
        next_b = fd.read(cls.peephole_bytes_length)
        if next_b == cls.no_peephole_bytes:
            return None
        elif next_b == cls.single_line_peephole:
            next_b = fd.read(cls.identifier_length)
            single_line_no = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
            return ProgramPeephole(program_container=program_container, start_line_no=single_line_no,
                                   end_line_no=single_line_no)
        else:
            print(next_b, next_b == cls.single_line_peephole, fd.read(-1))
            fd.seek(-cls.peephole_bytes_length, 1)
            next_b = fd.read(constant_pool.identifier_length)
            start_line_no = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
            next_b = fd.read(constant_pool.identifier_length)
            end_line_no = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
            return ProgramPeephole(program_container=program_container, start_line_no=start_line_no,
                                   end_line_no=end_line_no)

    @abstractmethod
    def cache(self, constant_pool) -> bytearray:
        """
        Tries to transform the abstract syntax tree of which this node is the root of into a corresponding byte array.
        This byte array can be used to restore an equivalent abstract syntax tree. This should enable a faster approach
        than parsing the program again. The constant pool stores all the constants of the program only one time and
        assigns them a value. This assigned value (reference) will be used in the byte code for each appearance of the
        constant. The pool can include names of methods and magic methods or string, numbers, etc.
        :param constant_pool: the constant pool that stores all the constants of the program only one time.
        :return: the byte array that represents the abstract syntax tree of which this node is the root of.
        """
        raise NotImplementedError('Cache-Method of %s is not implemented !' % self.__class__.__name__)

    @classmethod
    @abstractmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        """
        Constructs the abstract syntax tree from the byte array that has been the result of a previous cache procedure.
        If the construction of the abstract syntax tree fails due to a corrupted byte code, a ByteCodeCorruptedError
        will be thrown.
        :param fd: the file descriptor that points to the position, where the byte array representing the abstract
        syntax tree is located.
        :param constant_pool: the constant pool that shall be used to restore the constants.
        :param program_container: the program container that conatins at least the program string.
        :return: the abstract syntax tree constructed from the given syntax tree.
        """
        raise NotImplementedError('Construct-from-Cache-Method of %s is not implemented !' % cls.__name__)

    def __repr__(self) -> str:
        """
        Returns an infix representation of this abstract syntax tree.
        :return: an infix representation of this abstract syntax tree.
        """
        return '(%s %s)' % (self.__class__.__name__,
                            ('[' + ','.join([repr(k) if k is not None else '(None)' for k in
                                             self.children]) + ']') if self.children else  '')


class ASTLeftSideExpressionNode(object):
    """ This abstract class represents a left-side expressions to which a value can be assigned. """

    @abstractmethod
    def prepare(self, environment: Environment, program_stack: ProgramStack) -> ASTPrepareResult:
        """
        Prepares the given left-side expression node to which a value shall be assigned.
        :param environment: the environment in which context this ASTNode shall be prepared.
        :param program_stack: the program stack that represents the course of the program execution.
        :return: the result of the preparation.
        """
        raise NotImplementedError('Prepare-Method of %s not implemented !' % self.__class__.__name__)


class StatementsBlockNode(ASTNode):
    identifier = bytes([0x10])

    def __init__(self, statements=None, *args, **kwargs):
        super(StatementsBlockNode, self).__init__(*args, **kwargs)
        self.children += statements if statements is not None else []

    @property
    def empty(self):
        return bool(self.children)

    def append_statement(self, statement_node: ASTNode):
        self.children.append(statement_node)

    def prepend_statement(self, statement_node: ASTNode):
        self.children.insert(0, statement_node)

    def cache(self, constant_pool) -> bytearray:
        statements_block_bytes = bytearray(self.identifier)
        for child in self.children:
            statements_block_bytes.extend(child.cache(constant_pool))
        statements_block_bytes.extend(self.byte_separator)
        return statements_block_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container) -> ASTNode:
        statement_list = []
        next_b = fd.read(cls.identifier_length)
        while next_b and next_b != cls.byte_separator:
            if next_b not in byte_ast_dispatch:
                raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
            statement_list.append(byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container))
            next_b = fd.read(cls.identifier_length)
        return StatementsBlockNode(statement_list)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        for statement in self.children:
            result = statement.execute(environment, ProgramStack(program_stack, statement.peephole))
            if result.type == ASTExecutionResultType.void_ or result.type == ASTExecutionResultType.value_:
                continue
            else:
                return result
        return ASTExecutionResult(ASTExecutionResultType.void_, None)


byte_ast_dispatch[StatementsBlockNode.identifier] = StatementsBlockNode


def merge_statements(stmt_block_1: StatementsBlockNode, stmt_block_2: StatementsBlockNode):
    """
    Merges both given statements block; the order stays the same.
    :param stmt_block_1: the statement block that shall be merged with the other given statement block.
    :param stmt_block_2: the statement block that shall be merged with the other given statement block.
    :return: the result of the merge.
    """
    return StatementsBlockNode(stmt_block_1.children + stmt_block_2.children)


class AssignmentNode(ASTNode):
    identifier = bytes([0x11])

    def __init__(self, var_expr, value_expr, *args, **kwargs):
        super(AssignmentNode, self).__init__(*args, **kwargs)
        self.children.append(var_expr)
        self.children.append(value_expr)

    @property
    def variable_expression(self):
        return self.children[0]

    @property
    def value_expression(self):
        return self.children[1]

    def cache(self, constant_pool) -> bytearray:
        var_assignment_bytes = bytearray(self.identifier)
        var_assignment_bytes.extend(self.cache_peephole(constant_pool))
        var_assignment_bytes.extend(self.variable_expression.cache(constant_pool))
        var_assignment_bytes.extend(self.value_expression.cache(constant_pool))
        return var_assignment_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container) -> ASTNode:
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        var_expr = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        value_expr = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        return AssignmentNode(var_expr=var_expr, value_expr=value_expr, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        value_response = self.value_expression.execute(environment, program_stack)
        var_response = self.variable_expression.prepare(environment, program_stack)
        if var_response.type == ASTPrepareType.var_not_found_:
            environment.insert_variable(name=self.variable_expression.name, type=Variable, value=value_response.value)
        else:
            arguments = list(var_response.value[2:])
            arguments.append(value_response.value)
            try:
                getattr(*var_response.value[:2])(*arguments)
            except Exception as e:
                raise ITypeError(repr(e), program_stack=program_stack) from e
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self) -> str:
        return '(%s %s = %s)' % (
            self.__class__.__name__, repr(self.variable_expression), repr(self.value_expression))


byte_ast_dispatch[AssignmentNode.identifier] = AssignmentNode


class FunctionNode(ASTNode):
    """ This class represents a function definition with a non-empty trunk and none, one or more arguments. """
    identifier = bytes([0x12])

    def __init__(self, func_name: str, trunk: StatementsBlockNode, parameter_list=None, *args, **kwargs):
        super(FunctionNode, self).__init__(*args, **kwargs)
        self.environment = None
        self.children.append(parameter_list)
        self.children.append(trunk)
        self.function_name = func_name

    @property
    def parameter_list(self):
        return self.children[0]

    @property
    def trunk(self) -> StatementsBlockNode:
        return self.children[1]

    @property
    def documentation(self) -> str:
        if not self.trunk.empty:
            first_statement = self.trunk.children[0]
            if isinstance(first_statement, ConstantNode):
                first_stmt_const = first_statement.value
                return first_stmt_const if isinstance(first_stmt_const, str) else None
        return None

    def cache(self, constant_pool) -> bytearray:
        function_bytes = bytearray(self.identifier)
        function_bytes.extend(self.cache_peephole(constant_pool))
        function_bytes.extend(constant_pool.add(StringNode(self.function_name)))
        function_bytes.extend(self.trunk.cache(constant_pool))
        function_bytes.extend(
            self.byte_separator if self.parameter_list is None else self.parameter_list.cache(constant_pool))
        return function_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(constant_pool.identifier_length)
        # Read in function name
        function_name = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        # Trunk of function
        next_b = fd.read(cls.identifier_length)
        trunk = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        # Parameter list
        next_b = fd.read(cls.identifier_length)
        if not next_b or next_b == cls.byte_separator:
            return FunctionNode(func_name=function_name, trunk=trunk, peephole=peephole)
        elif next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        else:
            parameter_list = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
            return FunctionNode(func_name=function_name, trunk=trunk,
                                parameter_list=parameter_list, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        self.environment = environment
        total_parameters = dict()
        default_parameters = dict()
        for index, parameter_node in enumerate(self.parameter_list.execute(environment, program_stack).value):
            total_parameters[index] = parameter_node.parameter_name
            if parameter_node.default_expression is not None:
                default_parameters[parameter_node.parameter_name] = parameter_node.default_expression
        environment.insert_variable(name=self.function_name, type=Function,
                                    value=Function(name=self.function_name, ast_node=self.trunk,
                                                   environment=environment,
                                                   total_parameters=total_parameters,
                                                   default_parameters=default_parameters,
                                                   doc=self.documentation))
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self) -> str:
        return '(%s def ..%s.. %s { %s }' % (
            self.__class__.__name__, self.function_name, repr(self.parameter_list) if self.parameter_list else '',
            repr(self.trunk))


byte_ast_dispatch[FunctionNode.identifier] = FunctionNode


class ParameterNode(ASTNode):
    identifier = bytes([0x13])

    def __init__(self, parameter_name, default_expression=None, *args, **kwargs):
        super(ParameterNode, self).__init__(*args, **kwargs)
        self.children.append(parameter_name)
        self.children.append(default_expression)

    @property
    def parameter_name(self):
        return self.children[0]

    @property
    def default_expression(self):
        return self.children[1]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        pass

    def cache(self, constant_pool) -> bytearray:
        parameter_bytes = bytearray(self.identifier)
        parameter_bytes.extend(self.cache_peephole(constant_pool))
        parameter_bytes.extend(constant_pool.add(StringNode(self.parameter_name)))
        parameter_bytes.extend(
            self.byte_separator if self.default_expression is None else self.default_expression.cache(constant_pool))
        return parameter_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container) -> ASTNode:
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        # Parameter name
        length = constant_pool.constant_index_size(next_b)
        parameter_name = constant_pool.get(fd.read(length))
        next_b = fd.read(cls.identifier_length)
        if next_b == cls.byte_separator:
            return ParameterNode(parameter_name=parameter_name, peephole=peephole)
        elif next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        else:
            default_expr = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
            return ParameterNode(parameter_name=parameter_name, default_expression=default_expr, peephole=peephole)

    def __repr__(self) -> str:
        return '(%s %s %s)' % (
            self.__class__.__name__, self.parameter_name,
            '= %s' % repr(self.default_expression) if self.default_expression else '')


byte_ast_dispatch[ParameterNode.identifier] = ParameterNode


class ParameterListNode(ASTNode):
    identifier = bytes([0x14])

    def __init__(self, parameter_node=None, *args, **kwargs):
        super(ParameterListNode, self).__init__(*args, **kwargs)
        if parameter_node is not None:
            self.insert_parameter(parameter_node)

    def insert_parameter(self, parameter_node: ParameterNode):
        self.children.append(parameter_node)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType, self.children)

    def cache(self, constant_pool) -> bytearray:
        parameter_list_bytes = bytearray(self.identifier)
        #        parameter_list_bytes.extend(self.cache_peephole(constant_pool))
        for child in self.children:
            parameter_list_bytes.extend(child.cache(constant_pool))
        parameter_list_bytes.extend(self.byte_separator)
        return parameter_list_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        #        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        node = ParameterListNode()  # peephole=peephole)
        while next_b != cls.byte_separator:
            if next_b not in byte_ast_dispatch:
                raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
            node.insert_parameter(byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container))
            next_b = fd.read(cls.identifier_length)
        return node


byte_ast_dispatch[ParameterListNode.identifier] = ParameterListNode


class FunctionArgumentNode(ASTNode):
    identifier = bytes([0x15])

    def __init__(self, arg_expr, name=None, *args, **kwargs):
        super(FunctionArgumentNode, self).__init__(*args, **kwargs)
        self.children.append(arg_expr)
        self.children.append(name)

    @property
    def argument_expression(self):
        return self.children[0]

    @property
    def name(self):
        return self.children[1]

    def cache(self, constant_pool) -> bytearray:
        function_argument_bytes = bytearray(self.identifier)
        function_argument_bytes.extend(self.cache_peephole(constant_pool))
        function_argument_bytes.extend(self.argument_expression.cache(constant_pool))
        function_argument_bytes.extend(
            self.byte_separator if self.name is None else constant_pool.add(StringNode(self.name)))
        return function_argument_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        arg_expr = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b == cls.byte_separator:
            return FunctionArgumentNode(arg_expr=arg_expr, peephole=peephole)
        elif next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        else:
            length = constant_pool.constant_index_size(next_b)
            name = constant_pool.get(fd.read(length))
            return FunctionArgumentNode(name=name, arg_expr=arg_expr, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        # The default expressions are executed by the called function node.
        pass

    def __repr__(self):
        return '(%s %s %s)' % (
            self.__class__.__name__, '%s =' % self.name if self.name is not None else '', self.argument_expression)


byte_ast_dispatch[FunctionArgumentNode.identifier] = FunctionArgumentNode


class FunctionArgumentListNode(ASTNode):
    identifier = bytes([0x16])

    def __init__(self, func_arg: FunctionArgumentNode = None, *args, **kwargs):
        super(FunctionArgumentListNode, self).__init__(*args, **kwargs)
        self.fixed_arguments = list()
        self.named_arguments = dict()
        if func_arg is not None:
            self.insert_argument(func_arg)

    @property
    def total_arguments_count(self):
        return self.fixed_arguments_count + self.named_arguments_count

    @property
    def fixed_arguments_count(self):
        return len(self.fixed_arguments)

    @property
    def named_arguments_count(self):
        return len(self.named_arguments)

    def insert_argument(self, function_argument: FunctionArgumentNode):
        """
        Inserts the given argument into this argument list node.
        :param function_argument: the argument node that shall be appended to this argument list.
        """
        if function_argument.name is None:
            self.fixed_arguments.append(function_argument.argument_expression)
        else:
            self.named_arguments[function_argument.name] = function_argument.argument_expression
        self.children.append(function_argument)

    def cache(self, constant_pool) -> bytearray:
        function_argument_bytes = bytearray(self.identifier)
        function_argument_bytes.extend(self.cache_peephole(constant_pool))
        for child in self.children:
            function_argument_bytes.extend(child.cache(constant_pool))
        function_argument_bytes.extend(self.byte_separator)
        return function_argument_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        function_node = FunctionArgumentListNode(peephole=peephole)
        while next_b != cls.byte_separator:
            if next_b not in byte_ast_dispatch:
                raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
            function_node.insert_argument(
                byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container))
            next_b = fd.read(cls.identifier_length)
        return function_node

    def execute(self, environment: Environment, program_stack: ProgramStack):
        pass


byte_ast_dispatch[FunctionArgumentListNode.identifier] = FunctionArgumentListNode


class TestListNode(ASTNode):
    identifier = bytes([0x17])

    def __init__(self, test_node=None, *args, **kwargs):
        super(TestListNode, self).__init__(*args, **kwargs)
        if test_node is not None:
            self.append_test_node(test_node)

    def append_test_node(self, test_node: ASTNode):
        self.children.append(test_node)

    def cache(self, constant_pool) -> bytearray:
        testlist_bytes = bytearray(self.identifier)
        testlist_bytes.extend(self.cache_peephole(constant_pool))
        for child in self.children:
            testlist_bytes.extend(child.cache(constant_pool))
        testlist_bytes.extend(self.byte_separator)
        return testlist_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        test_list_node = TestListNode(peephole=peephole)
        while next_b != cls.byte_separator:
            if next_b not in byte_ast_dispatch:
                raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
            test_list_node.append_test_node(
                byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container))
            next_b = fd.read(cls.identifier_length)
        return test_list_node

    def execute(self, environment: Environment, program_stack: ProgramStack):
        value_list = list()
        for test_node in self.children:
            value_list.append(test_node.execute(environment, program_stack).value)
        return ASTExecutionResult(ASTExecutionResultType.value_, value_list)


byte_ast_dispatch[TestListNode.identifier] = TestListNode


class FunctionCallNode(ASTNode):
    identifier = bytes([0x18])

    def __init__(self, left_side_expression, argument_list=None, *args, **kwargs):
        super(FunctionCallNode, self).__init__(*args, **kwargs)
        self.children.append(left_side_expression)
        self.children.append(argument_list)

    @property
    def left_side_expression(self):
        return self.children[0]

    @property
    def argument_list(self):
        return self.children[1]

    def cache(self, constant_pool) -> bytearray:
        function_call_bytes = bytearray(self.identifier)
        function_call_bytes.extend(self.cache_peephole(constant_pool))
        function_call_bytes.extend(self.left_side_expression.cache(constant_pool))
        if self.argument_list is not None:
            function_call_bytes.extend(self.argument_list.cache(constant_pool))
        return function_call_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        left_side_expression = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b == cls.byte_separator:
            return FunctionCallNode(left_side_expression=left_side_expression, peephole=peephole)
        elif next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        else:
            argument_list = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
            return FunctionCallNode(left_side_expression=left_side_expression, argument_list=argument_list,
                                    peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        function = self.left_side_expression.execute(environment, program_stack).value
        if function is None or not hasattr(function, '__call__'):
            raise ITypeError('\'%s\' is not callable.' % function.__class__.__name__, program_stack)
        # Prepare the arguments for the call of the function.
        fixed_arguments = list()
        named_arguments = dict()
        if self.argument_list:
            # Compute arguments
            for fixed_arg in self.argument_list.fixed_arguments:
                fixed_arguments.append(fixed_arg.execute(environment, program_stack).value)
            for arg_name in self.argument_list.named_arguments:
                named_arguments[arg_name] = self.argument_list.named_arguments[arg_name].execute(environment,
                                                                                                 program_stack).value
        if type(function) == Function:
            return ASTExecutionResult(ASTExecutionResultType.value_, function.__call__(program_stack, *fixed_arguments,
                                                                                       **named_arguments))
        else:
            try:
                return ASTExecutionResult(ASTExecutionResultType.value_, function.__call__(*fixed_arguments,
                                                                                           **named_arguments))
            except Exception as e:
                raise InternalError(e, program_stack)

    def __repr__(self) -> str:
        return '(%s %s %s)' % (
            self.__class__.__name__, repr(self.left_side_expression),
            repr(self.argument_list) if self.argument_list else '()')


byte_ast_dispatch[FunctionCallNode.identifier] = FunctionCallNode


class PassNode(ASTNode):
    identifier = bytes([0x19])

    def __init__(self, *args, **kwargs):
        super(PassNode, self).__init__(*args, **kwargs)

    def cache(self, constant_pool) -> bytearray:
        pass_bytes = bytearray(self.identifier)
        pass_bytes.extend(self.cache_peephole(constant_pool))
        return pass_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        return PassNode(peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.void_, None)


byte_ast_dispatch[PassNode.identifier] = PassNode


class FlowControlNode(ASTNode):
    """ This class represents a node that maybe redirects the program flow. """

    def __init__(self, *args, **kwargs):
        super(FlowControlNode, self).__init__(*args, **kwargs)

    @abstractmethod
    def execute(self, environment: Environment, program_stack: ProgramStack):
        super(FlowControlNode, self).execute(environment)


class ContinueNode(FlowControlNode):
    identifier = bytes([0x1A])

    def __init__(self, *args, **kwargs):
        super(FlowControlNode, self).__init__(*args, **kwargs)

    def cache(self, constant_pool) -> bytearray:
        continue_bytes = bytearray(self.identifier)
        continue_bytes.extend(self.cache_peephole(constant_pool))
        return continue_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        return ContinueNode(peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.continue_, None)


byte_ast_dispatch[ContinueNode.identifier] = ContinueNode


class BreakNode(FlowControlNode):
    identifier = bytes([0x1B])

    def __init__(self, *args, **kwargs):
        super(BreakNode, self).__init__(*args, **kwargs)

    def cache(self, constant_pool) -> bytearray:
        break_bytes = bytearray(self.identifier)
        break_bytes.extend(self.cache_peephole(constant_pool))
        return break_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        return BreakNode(peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.break_, None)


byte_ast_dispatch[BreakNode.identifier] = BreakNode


class IfOperationNode(FlowControlNode):
    identifier = bytes([0x1C])

    def __init__(self, test, true_branch, else_branch=None, *args, **kwargs):
        super(IfOperationNode, self).__init__(*args, **kwargs)
        self.children.append(test)
        self.children.append(true_branch)
        self.children.append(else_branch)

    @property
    def test(self) -> ASTNode:
        return self.children[0]

    @property
    def true_branch(self) -> ASTNode:
        return self.children[1]

    @property
    def else_branch(self) -> ASTNode:
        return self.children[2]

    def cache(self, constant_pool) -> bytearray:
        if_operation_bytes = bytearray(self.identifier)
        if_operation_bytes.extend(self.cache_peephole(constant_pool))
        if_operation_bytes.extend(self.test.cache(constant_pool))
        if_operation_bytes.extend(self.true_branch.cache(constant_pool))
        if_operation_bytes.extend(
            self.byte_separator if self.else_branch is None else self.else_branch.cache(constant_pool))
        return if_operation_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        test = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        true_branch = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b == cls.byte_separator:
            return IfOperationNode(test=test, true_branch=true_branch, peephole=peephole)
        elif next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        else:
            else_branch = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
            return IfOperationNode(test=test, true_branch=true_branch, else_branch=else_branch, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        test_value = self.test.execute(environment, program_stack).value
        if test_value:
            return self.true_branch.execute(environment, program_stack)
        elif self.else_branch is not None:
            return self.else_branch.execute(environment, program_stack)
        else:
            return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self) -> str:
        return '(%s if %s then %s %s' % (self.__class__.__name__, repr(self.test), repr(self.true_branch),
                                         ' else %s' % repr(self.else_branch) if self.else_branch is not None else '')


byte_ast_dispatch[IfOperationNode.identifier] = IfOperationNode


class WhileOperationNode(FlowControlNode):
    identifier = bytes([0x1D])

    def __init__(self, test, trunk, else_branch=None, *args, **kwargs):
        super(WhileOperationNode, self).__init__(*args, **kwargs)
        self.children.append(test)
        self.children.append(trunk)
        self.children.append(else_branch)

    @property
    def test(self):
        return self.children[0]

    @property
    def trunk(self):
        return self.children[1]

    @property
    def else_branch(self):
        return self.children[2]

    def cache(self, constant_pool) -> bytearray:
        while_operation_bytes = bytearray(self.identifier)
        while_operation_bytes.extend(self.cache_peephole(constant_pool))
        while_operation_bytes.extend(self.test.cache(constant_pool))
        while_operation_bytes.extend(self.trunk.cache(constant_pool))
        while_operation_bytes.extend(
            self.byte_separator if self.else_branch is None else self.else_branch.cache(constant_pool))

        return while_operation_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        test = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        trunk = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        if next_b == cls.byte_separator:
            return WhileOperationNode(test=test, trunk=trunk, peephole=peephole)
        elif next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        else:
            else_branch = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
            return WhileOperationNode(test=test, trunk=trunk, else_branch=else_branch, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        while True:
            test_cond = self.test.execute(environment, program_stack).value
            if test_cond:
                trunk_response = self.trunk.execute(environment, program_stack)
                if trunk_response.type == ASTExecutionResultType.return_:
                    return trunk_response
                elif trunk_response.type == ASTExecutionResultType.continue_:
                    continue
                elif trunk_response.type == ASTExecutionResultType.break_:
                    return ASTExecutionResult(ASTExecutionResultType.void_, None)
            else:
                break
        if self.else_branch is not None:
            else_response = self.else_branch.execute(environment, program_stack)
            if else_response.type == ASTExecutionResultType.return_:
                return else_response
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self) -> str:
        return '(%s while %s do %s %s' % (self.__class__.__name__, repr(self.test), repr(self.trunk),
                                          ' else %s' % repr(self.else_branch) if self.else_branch is not None else '')


byte_ast_dispatch[WhileOperationNode.identifier] = WhileOperationNode


class ForOperationNode(ASTNode):
    identifier = bytes([0x1E])

    def __init__(self, variable_name: str, iterable_node: ASTNode, trunk: ASTNode, else_branch: ASTNode = None, *args,
                 **kwargs):
        super(ForOperationNode, self).__init__(*args, **kwargs)
        self.children.append(variable_name)
        self.children.append(iterable_node)
        self.children.append(trunk)
        self.children.append(else_branch)

    @property
    def variable_name(self):
        return self.children[0]

    @property
    def iterable_node(self):
        return self.children[1]

    @property
    def trunk(self):
        return self.children[2]

    @property
    def else_branch(self):
        return self.children[3]

    def cache(self, constant_pool) -> bytearray:
        for_operation_bytes = bytearray(self.identifier)
        for_operation_bytes.extend(self.cache_peephole(constant_pool))
        for_operation_bytes.extend(constant_pool.add(StringNode(self.variable_name)))
        for_operation_bytes.extend(self.iterable_node.cache(constant_pool))
        for_operation_bytes.extend(self.trunk.cache(constant_pool))
        for_operation_bytes.extend(
            self.byte_separator if self.else_branch is None else self.else_branch.cache(constant_pool))
        return for_operation_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        variable_name = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        iterable_node = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        trunk = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b == cls.byte_separator:
            return ForOperationNode(variable_name=variable_name, iterable_node=iterable_node, trunk=trunk,
                                    peephole=peephole)
        elif next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        else:
            else_branch = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
            return ForOperationNode(variable_name=variable_name, iterable_node=iterable_node, trunk=trunk,
                                    else_branch=else_branch, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        local_env = Environment(environment)
        iterable_obj = iter(self.iterable_node.execute(environment, program_stack).value)
        for entry in iterable_obj:
            local_env.insert_variable(name=self.variable_name, type=Variable, value=entry)
            trunk_response = self.trunk.execute(local_env, program_stack)
            if trunk_response.type == ASTExecutionResultType.return_:
                return trunk_response
            elif trunk_response.type == ASTExecutionResultType.continue_:
                continue
            elif trunk_response.type == ASTExecutionResultType.break_:
                return ASTExecutionResult(ASTExecutionResultType.void_, None)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self):
        return '(%s for %s in %s do %s %s)' % (
            self.__class__.__name__, self.variable_name, self.iterable_node, self.trunk,
            ' else %s' % repr(self.else_branch) if self.else_branch is not None else '')


byte_ast_dispatch[ForOperationNode.identifier] = ForOperationNode


class ReturnNode(FlowControlNode):
    identifier = bytes([0x1F])

    def __init__(self, return_expr=None, *args, **kwargs):
        super(ReturnNode, self).__init__(*args, **kwargs)
        self.children.append(return_expr)

    @property
    def return_expr(self): return self.children[0]

    def cache(self, constant_pool) -> bytearray:
        return_bytes = bytearray(self.identifier)
        return_bytes.extend(self.cache_peephole(constant_pool))
        return_bytes.extend(self.return_expr.cache(constant_pool))
        return return_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        return_expr = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        return ReturnNode(return_expr=return_expr, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        rtrn_value = self.return_expr.execute(environment,
                                              program_stack).value if self.return_expr is not None else None
        return ASTExecutionResult(ASTExecutionResultType.return_, rtrn_value)

    def __repr__(self) -> str:
        return '(%s return %s)' % (self.__class__.__name__, '' if self.return_expr is None else repr(self.return_expr))


byte_ast_dispatch[ReturnNode.identifier] = ReturnNode


class OperationNode(ASTNode):
    def __init__(self, op_name, magic_method, *args, **kwargs):
        super(OperationNode, self).__init__(*args, **kwargs)
        self.children.append(magic_method)
        self.operation_name = op_name

    @property
    def magic_method(self): return self.children[0]

    @abstractmethod
    def execute(self, environment: Environment, program_stack: ProgramStack):
        super(OperationNode, self).execute(environment)


class BinOperationNode(OperationNode):
    identifier = bytes([0x21])

    def __init__(self, op_name, magic_method, left, right, *args, **kwargs):
        """
        Initializes a new binary operator node with the given left and right child.
        :param op_name: the name of the operation.
        :param magic_method: the magic methods (right, left) for this operation.
        :param left: the left child of the binary operator.
        :param right: the right child of the binary operator.
        """
        super(BinOperationNode, self).__init__(op_name, magic_method, *args, **kwargs)
        self.children.append(left)
        self.children.append(right)

    @property
    def left_operand(self):
        return self.children[1]

    @property
    def right_operand(self):
        return self.children[2]

    def cache(self, constant_pool) -> bytearray:
        bin_operation_bytes = bytearray(self.identifier)
        bin_operation_bytes.extend(self.cache_peephole(constant_pool))
        bin_operation_bytes.extend(constant_pool.add(StringNode(self.magic_method)))
        bin_operation_bytes.extend(self.left_operand.cache(constant_pool))
        bin_operation_bytes.extend(self.right_operand.cache(constant_pool))
        return bin_operation_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        magic_method = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        left = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        right = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        return cls(op_name=magic_method, magic_method=magic_method, left=left, right=right, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        left_operand = self.left_operand.execute(environment, program_stack).value
        right_operand = self.right_operand.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  getattr(left_operand, self.magic_method)(right_operand))

    def __repr__(self) -> str:
        return '(%s %s %s %s)' % (
            self.__class__.__name__, repr(self.left_operand) if self.left_operand is not None else '(None)',
            self.operation_name,
            repr(self.right_operand) if self.right_operand is not None else '(None)')


byte_ast_dispatch[BinOperationNode.identifier] = BinOperationNode


class ComparisonOperationNode(BinOperationNode):
    identifier = bytes([0x22])

    def __init__(self, op_name, magic_method, left, right, *args, **kwargs):
        super(ComparisonOperationNode, self).__init__(op_name, magic_method, left, right, *args, **kwargs)


byte_ast_dispatch[ComparisonOperationNode.identifier] = ComparisonOperationNode


class BooleanBinOperationNode(BinOperationNode):
    identifier = bytes([0x23])

    def __init__(self, op_name, left, right, *args, **kwargs):
        super(BooleanBinOperationNode, self).__init__(op_name, None, left, right, *args, **kwargs)


byte_ast_dispatch[BooleanBinOperationNode.identifier] = BooleanBinOperationNode


class ANDOperationNode(BooleanBinOperationNode):
    identifier = bytes([0x24])

    def __init__(self, left, right, *args, **kwargs):
        super(BooleanBinOperationNode, self).__init__('and', None, left, right, *args, **kwargs)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        left_operand = self.left_operand.execute(environment, program_stack).value
        right_operand = self.right_operand.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, left_operand and right_operand)


byte_ast_dispatch[ANDOperationNode.identifier] = ANDOperationNode


class OROperationNode(BooleanBinOperationNode):
    identifier = bytes([0x25])

    def __init__(self, left, right, *args, **kwargs):
        super(BooleanBinOperationNode, self).__init__('or', None, left, right, *args, **kwargs)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        left_operand = self.left_operand.execute(environment, program_stack).value
        right_operand = self.right_operand.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, left_operand or right_operand)


byte_ast_dispatch[OROperationNode.identifier] = OROperationNode


class UnaryOperatorNode(OperationNode):
    identifier = bytes([0x26])

    def __init__(self, op_name, magic_method, node=None, *args, **kwargs):
        """
        Initializes a new unary operator node with the given node.
        :param op_name: the name of the unary operation.
        :param magic_method: the magic method for this operation
        :param node: the child of this operation node.
        """
        super(UnaryOperatorNode, self).__init__(op_name, magic_method, *args, **kwargs)
        self.children.append(node)

    @property
    def kid(self): return self.children[1]

    def cache(self, constant_pool) -> bytearray:
        unary_operation_bytes = bytearray(self.identifier)
        unary_operation_bytes.extend(self.cache_peephole(constant_pool))
        unary_operation_bytes.extend(constant_pool.add(StringNode(self.magic_method)))
        unary_operation_bytes.extend(self.kid.cache(constant_pool))
        return unary_operation_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        magic_method = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        kid = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        return UnaryOperatorNode(op_name=magic_method, magic_method=magic_method, node=kid, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        kid_value = self.kid.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, getattr(kid_value, self.magic_method)())

    def __repr__(self) -> str:
        return '(%s %s %s)' % (
            self.__class__.__name__, self.operation_name,
            repr(self.children[0]) if self.children[0] is not None else '(None)')


byte_ast_dispatch[UnaryOperatorNode.identifier] = UnaryOperatorNode


class UnaryBooleanOperationNode(UnaryOperatorNode):
    identifier = bytes([0x27])

    def __init__(self, op_name, node=None, *args, **kwargs):
        super(UnaryBooleanOperationNode, self).__init__(op_name, None, node, *args, **kwargs)


byte_ast_dispatch[UnaryBooleanOperationNode.identifier] = UnaryBooleanOperationNode


class NotOperationNode(UnaryBooleanOperationNode):
    identifier = bytes([0x28])

    def __init__(self, node=None, *args, **kwargs):
        super(NotOperationNode, self).__init__('not', node, *args, **kwargs)

    def cache(self, constant_pool) -> bytearray:
        not_op_bytes = bytearray(self.identifier)
        not_op_bytes.extend(self.cache_peephole(constant_pool))
        not_op_bytes.extend(self.kid.cache(constant_pool))
        return not_op_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier)
        if next_b in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        kid = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        return NotOperationNode(kid=kid, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        kid_value = self.kid.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, not kid_value)


byte_ast_dispatch[NotOperationNode.identifier] = NotOperationNode


class ConstantNode(ASTNode):
    def __init__(self, value, *args, **kwargs):
        super(ConstantNode, self).__init__(*args, **kwargs)
        self.children.append(value)

    @property
    def value(self): return self.children[0]

    @abstractmethod
    def to_bytes(self):
        """
        Converts the given constant in dependence of their type to the corresponding byte array and returns this array.
        :return: the byte array that results out of the conversion to bytes.
        """
        raise NotImplementedError('To-Byte-Method of %s is not implemented !' % self.__class__.__name__)

    @staticmethod
    @abstractmethod
    def from_bytes(fd):
        """
        Converts the read-in bytes of the given file descriptor into the constant value given the corresponding type
        of this node. If the read-in bytes does not fit the format of this constant, a ValueError is raised.
        :param fd: file descriptor pointing to the bytes that shall be converted to a constant.
        :return: the value corresponding to the given bytes array.
        """
        raise NotImplementedError('From-To-Method is not implemented !')

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.value_, self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, ConstantNode) and self.value == other.value

    def __repr__(self) -> str:
        return self.value.__repr__()


class NoneNode(ConstantNode):
    identifier = bytes([0x0A])

    def __init__(self, *args, **kwargs):
        super(NoneNode, self).__init__(None, *args, **kwargs)

    def to_bytes(self): return self.identifier

    @staticmethod
    def from_bytes(fd): return None

    def cache(self, constant_pool) -> bytearray:
        return bytearray(self.identifier)

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        return NoneNode()


byte_ast_dispatch[NoneNode.identifier] = NoneNode


class TrueNode(ConstantNode):
    identifier = bytes([0x0B])

    def __init__(self, *args, **kwargs):
        super(TrueNode, self).__init__(True, *args, **kwargs)

    def to_bytes(self): return self.identifier

    @staticmethod
    def from_bytes(fd): return True

    def cache(self, constant_pool) -> bytearray:
        return bytearray(self.identifier)

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        return TrueNode()


byte_ast_dispatch[TrueNode.identifier] = TrueNode


class FalseNode(ConstantNode):
    identifier = bytes([0x0C])

    def __init__(self, *args, **kwargs):
        super(FalseNode, self).__init__(False, *args, **kwargs)

    def to_bytes(self): return self.identifier

    @staticmethod
    def from_bytes(fd): return False

    def cache(self, constant_pool) -> bytearray:
        return bytearray(self.identifier)

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        return FalseNode()


byte_ast_dispatch[FalseNode.identifier] = FalseNode


class NumberNode(ConstantNode):
    identifier = bytes([0x0D])

    def __init__(self, value, *args, **kwargs):
        super(NumberNode, self).__init__(value, *args, **kwargs)

    def to_bytes(self):
        raw_int_queue = deque()
        neg = (self.value < 0)
        val = (abs(self.value) << 1) + neg
        potential_prefix = set(range(256))
        while True:
            next_i_byte = val & 0xFF
            raw_int_queue.appendleft(next_i_byte)
            if next_i_byte in potential_prefix:
                potential_prefix.remove(next_i_byte)
            val >>= 8
            if val == 0:
                break
        if not potential_prefix:
            raise ValueError(
                'Integer byte transformation failed due to the size of %d. It must be lower than 2^255.' % self.value)
        prefix = potential_prefix.pop()
        raw_int_queue.append(prefix)
        raw_int_queue.appendleft(prefix)
        # Construct integer byte.
        integer_bytes = bytearray(self.identifier)
        integer_bytes.extend(raw_int_queue)
        return integer_bytes

    @staticmethod
    def from_bytes(fd):
        int_bytes = bytearray()
        prefix = fd.read(1)
        next_b = fd.read(1)
        while next_b != prefix:
            int_bytes.extend(next_b)
            next_b = fd.read(1)
        value = int.from_bytes(reversed(int_bytes), byteorder='little')
        positive = value % 2 == 0
        value >>= 1
        return value if positive else -value

    def cache(self, constant_pool) -> bytearray:
        number_bytes = bytearray(self.identifier)
        number_bytes.extend(constant_pool.add(self))
        return number_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        next_b = fd.read(cls.identifier_length)
        number = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        return NumberNode(number)


byte_ast_dispatch[NumberNode.identifier] = NumberNode


class StringNode(ConstantNode):
    identifier = bytes([0x0E])

    def __init__(self, value, *args, **kwargs):
        assert value is not None
        super(StringNode, self).__init__(value, *args, **kwargs)

    def to_bytes(self):
        str_bytes = bytearray(self.identifier)
        str_bytes.extend(str.encode(self.value, 'utf-8'))
        str_bytes.extend(self.byte_separator)
        return str_bytes

    @staticmethod
    def from_bytes(fd):
        str_bytes = bytearray()
        while True:
            next_b = fd.read(1)
            if next_b == bytes([0x00]):
                break
            str_bytes.extend(next_b)
        return str_bytes.decode('utf-8')

    def cache(self, constant_pool) -> bytearray:
        string_bytes = bytearray(self.identifier)
        string_bytes.extend(constant_pool.add(self))
        return string_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        next_b = fd.read(cls.identifier_length)
        string = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        return StringNode(string)


byte_ast_dispatch[StringNode.identifier] = StringNode


class VariableNode(ASTNode, ASTLeftSideExpressionNode):
    identifier = bytes([0x29])

    def __init__(self, name, *args, **kwargs):
        super(VariableNode, self).__init__(*args, **kwargs)
        self.name = name

    def prepare(self, environment: Environment, program_stack: ProgramStack):
        variable = environment.get_variable(self.name)
        if variable is not None:
            return ASTPrepareResult(ASTPrepareType.var_found_, (variable, '__set_value__'))
        else:
            return ASTPrepareResult(ASTPrepareType.var_not_found_, variable)

    def cache(self, constant_pool) -> bytearray:
        var_bytes = bytearray(self.identifier)
        var_bytes.extend(constant_pool.add(StringNode(self.name)))
        return var_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        next_b = fd.read(cls.identifier_length)
        name = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        return VariableNode(name=name)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        variable = self.prepare(environment, program_stack)
        if variable.type == ASTPrepareType.var_not_found_:
            raise VariableError(error_message='Variable \'%s\' was not defined before.' % self.name,
                                program_stack=program_stack)
        return ASTExecutionResult(ASTExecutionResultType.value_, (variable.value)[0].value)

    def __repr__(self) -> str:
        return '(Variable %s)' % repr(self.name)


byte_ast_dispatch[VariableNode.identifier] = VariableNode


class PrefixNode(ASTNode):
    identifier = bytes([0x2A])

    def __init__(self, name, iri='', *args, **kwargs):
        super(PrefixNode, self).__init__(*args, **kwargs)
        self.name = name
        self.iri = iri

    def cache(self, constant_pool) -> bytearray:
        prefix_bytes = bytearray(self.identifier)
        prefix_bytes.extend(self.cache_peephole(constant_pool))
        prefix_bytes.extend(constant_pool.add(self.name))
        prefix_bytes.extend(constant_pool.add(self.iri))
        return prefix_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        name = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        next_b = fd.read(cls.identifier_length)
        iri = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        return PrefixNode(name=name, iri=iri, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        if self.name is not None:
            environment.insert_prefix(name=self.name, iri=self.iri)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self):
        return '(%s %s : %s)' % (self.__class__.__name__, self.name, self.iri)


byte_ast_dispatch[PrefixNode.identifier] = PrefixNode


class ResourceNode(ASTNode):
    identifier = bytes([0x2B])

    def __init__(self, iri, prefix_name=None, *args, **kwargs):
        super(ResourceNode, self).__init__(*args, **kwargs)
        self.prefix_name = prefix_name
        self.iri = iri

    def cache(self, constant_pool) -> bytearray:
        resource_bytes = bytearray(self.identifier)
        resource_bytes.extend(self.cache_peephole(constant_pool))
        resource_bytes.extend(constant_pool.add(self.iri))
        resource_bytes.extend(self.byte_separator if self.iri is None else constant_pool.add(self.prefix_name))
        return resource_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        peephole = cls.construct_peephole_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier)
        iri = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        if next_b == cls.byte_separator:
            return ResourceNode(iri=iri)
        next_b = fd.read(cls.identifier)
        prefix_name = constant_pool.get(fd.read(constant_pool.constant_index_size(next_b)))
        return ResourceNode(iri=iri, prefix_name=prefix_name, peephole=peephole)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        pref = ''
        if self.prefix_name is not None:
            pref = environment.get_prefix(self.prefix_name)
            if pref is None:
                raise ValueError('Prefix %s was not declared.' % self.prefix_name)
        return ASTExecutionResult(ASTExecutionResultType.value_, resource(pref + self.iri))

    def __repr__(self):
        return '(%s < %s >)' % (
            self.__class__.__name__,
            '%s : %s' % (self.prefix_name, self.iri) if self.prefix_name is not None else self.iri)


byte_ast_dispatch[ResourceNode.identifier] = ResourceNode


class TripleNode(ASTNode):
    identifier = bytes([0x2C])

    def __init__(self, subject, predicate, object, *args, **kwargs):
        super(TripleNode, self).__init__(*args, **kwargs)
        self.children.append(subject)
        self.children.append(predicate)
        self.children.append(object)

    @property
    def subject(self):
        return self.children[0]

    @property
    def predicate(self):
        return self.children[1]

    @property
    def object(self):
        return self.children[2]

    def cache(self, constant_pool) -> bytearray:
        triple_bytes = bytearray(self.identifier)
        triple_bytes.extend(self.subject.cache(constant_pool))
        triple_bytes.extend(self.predicate.cache(constant_pool))
        triple_bytes.extend(self.object.cache(constant_pool))
        return triple_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        next_b = fd.read(cls.identifier_length)
        if next_b is not byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        subject = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b is not byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        predicate = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        if next_b is not byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        object = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        return TripleNode(subject=subject, predicate=predicate, object=object)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        subj = self.subject.execute(environment).value
        pred = self.predicate.execute(environment).value
        obj = self.object.execute(environment).value
        return ASTExecutionResult(ASTExecutionResultType.value_, triple((subj, pred, obj)))

    def __repr__(self):
        return '(%s %s %s)' % (self.subject, self.predicate, self.object)


byte_ast_dispatch[TripleNode.identifier] = TripleNode


class AtomListNode(ASTNode):
    identifier = bytes([0x2D])

    def __init__(self, atom_node=None, *args, **kwargs):
        super(AtomListNode, self).__init__(*args, **kwargs)
        if atom_node is not None:
            self.append_atom_node(atom_node)

    def append_atom_node(self, atom_node: ASTNode):
        """
        Appends the given test node at the end of the list.
        :param atom_node: the test node that shall be appended at the end of the test list.
        """
        self.children.append(atom_node)

    def cache(self, constant_pool) -> bytearray:
        atom_list_bytes = bytearray(self.identifier)
        for child in self.children:
            atom_list_bytes.extend(child.cache(constant_pool))
        atom_list_bytes.extend(self.byte_separator)
        return atom_list_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        next_b = fd.read(cls.identifier_length)
        atom_list_node = AtomListNode()
        while next_b != cls.byte_separator:
            if next_b is not byte_ast_dispatch:
                raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
            atom_list_node.append_atom_node(
                byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container))
            next_b = fd.read(cls.identifier_length)
        return atom_list_node

    def execute(self, environment: Environment, program_stack: ProgramStack):
        value_list = list()
        for atom_node in self.children:
            value_list.append(atom_node.execute(environment).value)
        return ASTExecutionResult(ASTExecutionResultType.value_, value_list)


byte_ast_dispatch[AtomListNode.identifier] = AtomListNode


class GraphNode(ASTNode):
    identifier = bytes([0x2E])

    def __init__(self, graph_construction_node=None, *args, **kwargs):
        super(GraphNode, self).__init__(*args, **kwargs)
        self.children.append(graph_construction_node)

    @property
    def construction_node(self):
        return self.children[0]

    def cache(self, constant_pool) -> bytearray:
        graph_bytes = bytearray(self.identifier)
        graph_bytes.extend(
            self.byte_separator if self.construction_node is None else self.construction_node.cache(constant_pool))
        return graph_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        next_b = fd.read(cls.identifier_length)
        if next_b == cls.byte_separator:
            return GraphNode()
        elif next_b is not byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        else:
            graph_construction_node = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool,
                                                                                     program_container)
            return GraphNode(graph_construction_node=graph_construction_node)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  graph(self.construction_node.execute(
                                      environment).value) if self.construction_node is not None else graph())


byte_ast_dispatch[GraphNode.identifier] = GraphNode


class ListNode(ASTNode):
    identifier = bytes([0x2E])

    def __init__(self, list_construction_node=None, *args, **kwargs):
        super(ListNode, self).__init__(*args, **kwargs)
        self.children.append(list_construction_node)

    @property
    def list_construction_node(self):
        return self.children[0]

    def cache(self, constant_pool) -> bytearray:
        list_bytes = bytearray(self.identifier)
        list_bytes.extend(
            self.byte_separator if self.list_construction_node is None else self.list_construction_node.cache(
                constant_pool))
        return list_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        next_b = fd.read(cls.identifier_length)
        if next_b == cls.byte_separator:
            return ListNode()
        elif next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        else:
            list_construction_node = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool,
                                                                                    program_container)
            return ListNode(list_construction_node=list_construction_node)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        if self.list_construction_node is None:
            return ASTExecutionResult(ASTExecutionResultType.value_, list())
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  self.list_construction_node.execute(environment, program_stack).value)


byte_ast_dispatch[ListNode.identifier] = ListNode


class SubscriptNode(ASTNode, ASTLeftSideExpressionNode):
    """ This class presents a node that represents the subscription of container objects. """

    identifier = bytes([0x2F])

    def __init__(self, container_node: ASTNode, subscript_node: ASTNode, *args, **kwargs):
        super(SubscriptNode, self).__init__(*args, **kwargs)
        self.children.append(container_node)
        self.children.append(subscript_node)

    @property
    def container_node(self):
        return self.children[0]

    @property
    def subscript_node(self):
        return self.children[1]

    def cache(self, constant_pool) -> bytearray:
        subscript_bytes = bytearray(self.identifier)
        subscript_bytes.extend(self.container_node.cache(constant_pool))
        subscript_bytes.extend(self.subscript_node.cache(constant_pool))
        return subscript_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        container_node = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        if next_b not in byte_ast_dispatch:
            raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
        subscript_node = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        return SubscriptNode(container_node=container_node, subscript_node=subscript_node)

    def prepare(self, environment: Environment, program_stack: ProgramStack) -> ASTPrepareResult:
        container_value = self.container_node.execute(environment, program_stack).value
        if container_value is None or not hasattr(container_value, '__getitem__'):
            raise ITypeError('\'%s\' is not a container.' % container_value.__class__.__name__, program_stack)
        index_value = self.subscript_node.execute(environment, program_stack).value
        return ASTPrepareResult(ASTPrepareType.var_found_, (container_value, '__setitem__', index_value))

    def execute(self, environment: Environment, program_stack: ProgramStack):
        container_value = self.container_node.execute(environment, program_stack).value
        if container_value is None or not hasattr(container_value, '__getitem__'):
            raise ITypeError('\'%s\' is not a container.' % container_value.__class__.__name__, program_stack)
        index_value = self.subscript_node.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, container_value.__getitem__(index_value))


byte_ast_dispatch[SubscriptNode.identifier] = SubscriptNode


class SliceNode(ASTNode):
    identifier = bytes([0x30])

    def __init__(self, lower_node=None, upper_node=None, step_node=None, *args, **kwargs):
        super(SliceNode, self).__init__(*args, **kwargs)
        self.children.append(lower_node)
        self.children.append(upper_node)
        self.children.append(step_node)

    @property
    def lower_node(self):
        return self.children[0]

    @property
    def upper_node(self):
        return self.children[1]

    @property
    def step_node(self):
        return self.children[2]

    def cache(self, constant_pool) -> bytearray:
        slice_bytes = bytearray(self.identifier)
        slice_bytes.extend(self.byte_separator if self.lower_node is None else self.lower_node.cache(constant_pool))
        slice_bytes.extend(self.byte_separator if self.upper_node is None else self.upper_node.cache(constant_pool))
        slice_bytes.extend(self.byte_separator if self.step_node is None else self.step_node.cache(constant_pool))
        return slice_bytes

    @classmethod
    def construct_from_cache(cls, fd, constant_pool, program_container):
        next_b = fd.read(cls.identifier_length)
        lower_node = None
        if next_b != cls.byte_separator:
            if next_b not in byte_ast_dispatch:
                raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
            lower_node = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        upper_node = None
        if next_b != cls.byte_separator:
            if next_b not in byte_ast_dispatch:
                raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
            upper_node = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        next_b = fd.read(cls.identifier_length)
        step_node = None
        if next_b != cls.byte_separator:
            if next_b not in byte_ast_dispatch:
                raise IntermediateCodeCorruptedError('Byte-Code is corrupted at position %d.' % fd.tell())
            step_node = byte_ast_dispatch[next_b].construct_from_cache(fd, constant_pool, program_container)
        return SliceNode(lower_node=lower_node, upper_node=upper_node, step_node=step_node)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        lower = self.lower_node.execute(environment, program_stack).value if self.lower_node is not None else None
        upper = self.upper_node.execute(environment, program_stack).value if self.upper_node is not None else None
        step = self.step_node.execute(environment, program_stack).value if self.step_node is not None else None
        return ASTExecutionResult(ASTExecutionResultType.value_, slice(lower, upper, step))


byte_ast_dispatch[SliceNode.identifier] = SliceNode
