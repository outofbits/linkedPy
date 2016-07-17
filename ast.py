# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

from enum import Enum

from abc import abstractmethod
from collections import namedtuple, deque

from env import Environment, Function, Variable, ProgramPeephole, ProgramStack
from exception import VariableError, InternalError, TypeError as ITypeError
from linkedtypes import resource, triple, graph


class ASTExecutionResultType(Enum):
    void_ = 1
    value_ = 2
    return_ = 3
    continue_ = 4
    break_ = 5


class ASTPrepareType(Enum):
    accessible_ = 1
    not_found_ = 2


ASTExecutionResult = namedtuple('ASTExecutionResult', ['type', 'value'])
ASTPrepareResult = namedtuple('ASTPrepareResult', ['type', 'value'])


def merge_statements(stmt_block_1, stmt_block_2):
    """
    Merges both given statements block; the order stays the same.
    :param stmt_block_1: the statement block that shall be merged with the other given statement block.
    :param stmt_block_2: the statement block that shall be merged with the other given statement block.
    :return: the result of the merge.
    """
    return StatementsBlockNode(stmt_block_1.children + stmt_block_2.children)


class ASTNode(object):
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

    def execute(self, environment: Environment, program_stack: ProgramStack):
        for statement in self.children:
            result = statement.execute(environment, ProgramStack(program_stack, statement.peephole))
            if result.type == ASTExecutionResultType.void_ or result.type == ASTExecutionResultType.value_:
                continue
            else:
                return result
        return ASTExecutionResult(ASTExecutionResultType.void_, None)


class VariableAssignmentNode(ASTNode):
    def __init__(self, var_expr, value_expr, *args, **kwargs):
        super(VariableAssignmentNode, self).__init__(*args, **kwargs)
        self.children.append(var_expr)
        self.children.append(value_expr)

    @property
    def variable_expression(self):
        return self.children[0]

    @property
    def value_expression(self):
        return self.children[1]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        value_response = self.value_expression.execute(environment, program_stack)
        var_response = self.variable_expression.prepare(environment, program_stack)
        if var_response.type == ASTPrepareType.not_found_:
            environment.insert_variable(name=self.variable_expression.name, type=Variable, value=value_response.value)
        else:
            var_response.value.change_value(value_response.value)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self) -> str:
        return '(%s %s = %s)' % (
            self.__class__.__name__, repr(self.variable_expression), repr(self.value_expression))


class FunctionNode(ASTNode):
    """ This class represents a function definition with a non-empty trunk and none, one or more arguments. """

    def __init__(self, func_name: str, trunk: StatementsBlockNode, parameter_list=None, *args, **kwargs):
        super(FunctionNode, self).__init__(*args, **kwargs)
        self.environment = None
        self.children.append(func_name)
        self.children.append(parameter_list)
        self.children.append(trunk)

    @property
    def function_name(self) -> str:
        return self.children[0]

    @property
    def parameter_list(self):
        return self.children[1]

    @property
    def trunk(self) -> StatementsBlockNode:
        return self.children[2]

    @property
    def documentation(self) -> str:
        if not self.trunk.empty:
            first_statement = self.trunk.children[0]
            if isinstance(first_statement, ConstantNode):
                first_stmt_const = first_statement.value
                return first_stmt_const if isinstance(first_stmt_const, str) else None
        return None

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


class ParameterNode(ASTNode):
    def __init__(self, parameter_name, default_expression, *args, **kwargs):
        super(ParameterNode, self).__init__(*args, **kwargs)
        self.children.append(parameter_name)
        self.children.append(default_expression)

    @property
    def parameter_name(self): return self.children[0]

    @property
    def default_expression(self): return self.children[1]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        pass

    def __repr__(self) -> str:
        return '(%s %s %s)' % (
            self.__class__.__name__, self.parameter_name,
            '= %s' % repr(self.default_expression) if self.default_expression else '')


class ParameterListNode(ASTNode):
    def __init__(self, parameter_node=None, *args, **kwargs):
        super(ParameterListNode, self).__init__(*args, **kwargs)
        if parameter_node is not None:
            self.insert_parameter(parameter_node)

    def insert_parameter(self, parameter_node: ParameterNode):
        self.children.append(parameter_node)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType, self.children)


class FunctionArgumentNode(ASTNode):
    def __init__(self, arg_expr, name=None, *args, **kwargs):
        super(FunctionArgumentNode, self).__init__(*args, **kwargs)
        self.children.append(arg_expr)
        self.children.append(name)

    @property
    def argument_expression(self): return self.children[0]

    @property
    def name(self): return self.children[1]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        # The default expressions are executed by the called function node.
        pass

    def __repr__(self):
        return '(%s %s %s)' % (
            self.__class__.__name__, '%s =' % self.name if self.name is not None else '', self.argument_expression)


class FunctionArgumentListNode(ASTNode):
    def __init__(self, func_arg: FunctionArgumentNode, *args, **kwargs):
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

    def execute(self, environment: Environment, program_stack: ProgramStack):
        pass


class TestListNode(ASTNode):
    def __init__(self, test_node=None, *args, **kwargs):
        super(TestListNode, self).__init__(*args, **kwargs)
        if test_node is not None:
            self.append_test_node(test_node)

    def append_test_node(self, test_node: ASTNode):
        self.children.append(test_node)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        value_list = list()
        for test_node in self.children:
            value_list.append(test_node.execute(environment, program_stack).value)
        return ASTExecutionResult(ASTExecutionResultType.value_, value_list)


class FunctionCallNode(ASTNode):
    """ This class represents the call of a function with the given amount of arguments. """

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


class PassNode(ASTNode):
    def __init__(self, *args, **kwargs):
        super(PassNode, self).__init__(*args, **kwargs)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.void_, None)


class PrintNode(ASTNode):
    def __init__(self, print_expression, *args, **kwargs):
        super(PrintNode, self).__init__(*args, **kwargs)
        self.children.append(print_expression)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        print(self.children[0].execute(environment).value)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)


class FlowControlNode(ASTNode):
    """ This class represents a node that maybe redirects the program flow. """

    def __init__(self, *args, **kwargs):
        super(FlowControlNode, self).__init__(*args, **kwargs)

    @abstractmethod
    def execute(self, environment: Environment, program_stack: ProgramStack):
        super(FlowControlNode, self).execute(environment)


class ContinueNode(FlowControlNode):
    def __init__(self, *args, **kwargs):
        super(FlowControlNode, self).__init__(*args, **kwargs)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.continue_, None)


class BreakNode(FlowControlNode):
    def __init__(self, *args, **kwargs):
        super(BreakNode, self).__init__(*args, **kwargs)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.break_, None)


class IfOperationNode(FlowControlNode):
    def __init__(self, test, true_branch, else_branch=None, *args, **kwargs):
        super(IfOperationNode, self).__init__(*args, **kwargs)
        self.children.append(test)
        self.children.append(true_branch)
        self.children.append(else_branch)

    @property
    def test(self) -> ASTNode:
        return self.children[0]

    @property
    def true_branch(self):
        return self.children[1]

    @property
    def else_branch(self) -> ASTNode:
        return self.children[2]

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


class WhileOperationNode(FlowControlNode):
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


class ForOperationNode(ASTNode):
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


class ReturnNode(FlowControlNode):
    def __init__(self, return_expr=None, *args, **kwargs):
        super(ReturnNode, self).__init__(*args, **kwargs)
        self.children.append(return_expr)

    @property
    def return_expr(self): return self.children[0]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        rtrn_value = self.return_expr.execute(environment,
                                              program_stack).value if self.return_expr is not None else None
        return ASTExecutionResult(ASTExecutionResultType.return_, rtrn_value)

    def __repr__(self) -> str:
        return '(%s return %s)' % (self.__class__.__name__, '' if self.return_expr is None else repr(self.return_expr))


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
    def __init__(self, op_name, magic_method, left=None, right=None, *args, **kwargs):
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
    def left_operand(self): return self.children[1]

    @property
    def right_operand(self): return self.children[2]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        left_operand = self.left_operand.execute(environment, program_stack).value
        right_operand = self.right_operand.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  getattr(left_operand, self.magic_method)(right_operand))

    def __repr__(self) -> str:
        return '(%s %s %s %s)' % (
            self.__class__.__name__, repr(self.children[0]) if self.children[0] is not None else '(None)',
            self.operation_name,
            repr(self.children[1]) if self.children[1] is not None else '(None)')


class ComparisonOperationNode(BinOperationNode):
    def __init__(self, op_name, magic_method, left=None, right=None, *args, **kwargs):
        super(ComparisonOperationNode, self).__init__(op_name, magic_method, left, right, *args, **kwargs)


class BooleanBinOperationNode(BinOperationNode):
    def __init__(self, op_name, left=None, right=None, *args, **kwargs):
        super(BooleanBinOperationNode, self).__init__(op_name, None, left, right, *args, **kwargs)


class ANDOperationNode(BooleanBinOperationNode):
    def __init__(self, left=None, right=None, *args, **kwargs):
        super(BooleanBinOperationNode, self).__init__('and', None, left, right, *args, **kwargs)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        left_operand = self.left_operand.execute(environment, program_stack).value
        right_operand = self.right_operand.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, left_operand and right_operand)


class OROperationNode(BooleanBinOperationNode):
    def __init__(self, left=None, right=None, *args, **kwargs):
        super(BooleanBinOperationNode, self).__init__('or', None, left, right, *args, **kwargs)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        left_operand = self.left_operand.execute(environment, program_stack).value
        right_operand = self.right_operand.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, left_operand or right_operand)


class UnaryOperatorNode(OperationNode):
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

    def execute(self, environment: Environment, program_stack: ProgramStack):
        kid_value = self.kid.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, getattr(kid_value, self.magic_method)())

    def __repr__(self) -> str:
        return '(%s %s %s)' % (
            self.__class__.__name__, self.operation_name,
            repr(self.children[0]) if self.children[0] is not None else '(None)')


class UnaryBooleanOperationNode(UnaryOperatorNode):
    def __init__(self, op_name, node=None, *args, **kwargs):
        super(UnaryBooleanOperationNode, self).__init__(op_name, None, node, *args, **kwargs)


class NotOperationNode(UnaryBooleanOperationNode):
    def __init__(self, node=None, *args, **kwargs):
        super(NotOperationNode, self).__init__('not', node, *args, **kwargs)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        kid_value = self.kid.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, not kid_value)


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
        raise NotImplementedError('From-To-Method of %s is not implemented !' % self.__class__.__name__)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.value_, self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, ConstantNode) and self.value == other.value

    def __repr__(self) -> str:
        return self.value.__repr__()


class NoneNode(ConstantNode):
    identifier = bytes([0x00, 0x0A])

    def __init__(self, *args, **kwargs):
        super(NoneNode, self).__init__(None, *args, **kwargs)

    def to_bytes(self): return self.identifier

    @staticmethod
    def from_bytes(fd): return None


class TrueNode(ConstantNode):
    identifier = bytes([0x00, 0x0B])

    def __init__(self, *args, **kwargs):
        super(TrueNode, self).__init__(True, *args, **kwargs)

    def to_bytes(self): return self.identifier

    @staticmethod
    def from_bytes(fd): return True


class FalseNode(ConstantNode):
    identifier = bytes([0x00, 0x0C])

    def __init__(self, *args, **kwargs):
        super(FalseNode, self).__init__(False, *args, **kwargs)

    def to_bytes(self): return self.identifier

    @staticmethod
    def from_bytes(fd): return False


class NumberNode(ConstantNode):
    identifier = bytes([0x00, 0x0D])

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
            potential_prefix.remove(next_i_byte)
            val >>= 16
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
        value = int.from_bytes(int_bytes, byteorder='little')
        positive = value % 2 == 0
        value >>= 1
        return value if positive else -value


class StringNode(ConstantNode):
    identifier = bytes([0x00, 0x0E])

    def __init__(self, value, *args, **kwargs):
        super(StringNode, self).__init__(value, *args, **kwargs)

    def to_bytes(self):
        str_bytes = bytearray(self.identifier)
        str_bytes.extend(str.encode(self.value, 'utf-8'))
        str_bytes.append(0x00)
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


class VariableNode(ASTNode, ASTLeftSideExpressionNode):
    def __init__(self, name, *args, **kwargs):
        super(VariableNode, self).__init__(*args, **kwargs)
        self.name = name

    def prepare(self, environment: Environment, program_stack: ProgramStack):
        var_description = environment.get_variable(self.name)
        return ASTExecutionResult(
            ASTPrepareType.accessible_ if var_description is not None else ASTPrepareType.not_found_, var_description)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        var_description = self.prepare(environment, program_stack)
        if var_description.type == ASTPrepareType.not_found_:
            raise VariableError(error_message='Variable \'%s\' was not defined before.' % self.name,
                                program_stack=program_stack)
        return ASTExecutionResult(ASTExecutionResultType.value_, (var_description.value).value)

    def __repr__(self) -> str:
        return '(Variable %s)' % repr(self.name)


class PrefixNode(ASTNode):
    def __init__(self, name, iri='', *args, **kwargs):
        super(PrefixNode, self).__init__(*args, **kwargs)
        self.name = name
        self.iri = iri

    def execute(self, environment: Environment, program_stack: ProgramStack):
        if self.name is not None:
            environment.insert_prefix(name=self.name, iri=self.iri)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self):
        return '(%s %s : %s)' % (self.__class__.__name__, self.name, self.iri)


class ResourceNode(ASTNode):
    def __init__(self, iri, prefix_name=None, *args, **kwargs):
        super(ResourceNode, self).__init__(*args, **kwargs)
        self.prefix_name = prefix_name
        self.iri = iri

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


class TripleNode(ASTNode):
    def __init__(self, subject, predicate, object, *args, **kwargs):
        super(TripleNode, self).__init__(*args, **kwargs)
        self.subject = subject
        self.predicate = predicate
        self.object = object

    def execute(self, environment: Environment, program_stack: ProgramStack):
        subj = self.subject.execute(environment).value
        pred = self.predicate.execute(environment).value
        obj = self.object.execute(environment).value
        return ASTExecutionResult(ASTExecutionResultType.value_, triple((subj, pred, obj)))

    def __repr__(self):
        return '(%s %s %s)' % (self.subject, self.predicate, self.object)


class AtomListNode(ASTNode):
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

    def execute(self, environment: Environment, program_stack: ProgramStack):
        value_list = list()
        for atom_node in self.children:
            value_list.append(atom_node.execute(environment).value)
        return ASTExecutionResult(ASTExecutionResultType.value_, value_list)


class GraphNode(ASTNode):
    def __init__(self, graph_construction_node=None, *args, **kwargs):
        super(GraphNode, self).__init__(*args, **kwargs)
        self.children.append(graph_construction_node)

    @property
    def construction_node(self): return self.children[0]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  graph(self.construction_node.execute(
                                      environment).value) if self.construction_node is not None else graph())


class ListNode(ASTNode):
    def __init__(self, list_construction_node, *args, **kwargs):
        super(ListNode, self).__init__(*args, **kwargs)
        self.children.append(list_construction_node)

    @property
    def list_construction_node(self): return self.children[0]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        if self.list_construction_node is None:
            return ASTExecutionResult(ASTExecutionResultType.value_, list())
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  self.list_construction_node.execute(environment, program_stack).value)


class SubscriptNode(ASTNode):
    """ This class presents a node that represents the subscription of container objects. """

    def __init__(self, container_node: ASTNode, subscript_node: ASTNode, *args, **kwargs):
        super(SubscriptNode, self).__init__(*args, **kwargs)
        self.children.append(container_node)
        self.children.append(subscript_node)

    @property
    def container_node(self):
        return self.children[0]

    @property
    def index_node(self):
        return self.children[1]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        container_value = self.container_node.execute(environment, program_stack).value
        if container_value is None or not hasattr(container_value, '__getitem__'):
            raise ITypeError('\'%s\' is not a container.' % container_value.__class__.__name__, program_stack)
        index_value = self.index_node.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, container_value.__getitem__(index_value))


class SliceNode(ASTNode):
    def __init__(self, lower_node=None, upper_node=None, step_node=None, *args, **kwargs):
        super(SliceNode, self).__init__(*args, **kwargs)
        self.children.append(lower_node)
        self.children.append(upper_node)
        self.children.append(step_node)

    @property
    def lower_node(self): return self.children[0]

    @property
    def upper_node(self): return self.children[1]

    @property
    def step_node(self): return self.children[2]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        lower = self.lower_node.execute(environment, program_stack).value if self.lower_node is not None else None
        upper = self.upper_node.execute(environment, program_stack).value if self.upper_node is not None else None
        step = self.step_node.execute(environment, program_stack).value if self.step_node is not None else None
        return ASTExecutionResult(ASTExecutionResultType.value_, slice(lower, upper, step))