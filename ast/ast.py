# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

from .exception import ExecutionError, VariableError, InternalError, TypeError as ITypeError
from .env import Environment, Function, Variable, ProgramPeephole, ProgramStack
from datatypes.linkedtypes import resource, triple, graph
from abc import abstractmethod
from enum import Enum
from collections import namedtuple, deque


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


class ASTNode(object):
    """ This class represents a node of an abstract syntax tree. """

    def __init__(self, peephole: ProgramPeephole = None):
        self.child = []
        self.peephole = peephole

    @property
    def children(self):
        return self.child

    @abstractmethod
    def execute(self, environment: Environment, program_stack: ProgramStack) -> ASTExecutionResult:
        """
        Executes this ASTNode and returns the corresponding result of the execution.
        :param environment: the environment in which context this ASTNode shall be executed.
        :param program_stack: the program stack that represents the course of the program execution.
        :return: the result of the execution.
        """
        raise NotImplementedError('Execute-Method of %s not implemented !' % self.__class__.__name__)

    def __repr__(self) -> str:
        """
        Returns an infix representation of this abstract syntax tree.
        :return: an infix representation of this abstract syntax tree.
        """
        return '(%s %s)' % (self.__class__.__name__,
                            ('[' + ','.join([repr(k) if k is not None else '(None)' for k in
                                             self.child]) + ']') if self.child else  '')


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
    """ This class represents an abstract syntax tree that hold a sequence of child nodes that shall be executed in the
        given order.
    """

    def __init__(self, statements=None, *args, **kwargs):
        super(StatementsBlockNode, self).__init__(*args, **kwargs)
        self.child += statements if statements is not None else []

    @staticmethod
    def merge(stmt_block_1, stmt_block_2) -> ASTNode:
        """
        Merges both given statements block; the order stays the same.
        :param stmt_block_1: the statement block that shall be merged with the other given statement block.
        :param stmt_block_2: the statement block that shall be merged with the other given statement block.
        :return: the result of the merge.
        """
        return StatementsBlockNode(stmt_block_1.children + stmt_block_2.children)

    def append_statement(self, statement_node: ASTNode):
        """
        Appends the given statement node at the end of the block.
        :param statement_node: the statement node that shall be appended at the end of the statement block.
        """
        self.child.append(statement_node)

    def insert_statement_at(self, statement_node: ASTNode, n: int = None):
        """
        Inserts the given statement node at the given position. If the given position n is None, the node will be
        appended.
        :param statement_node: the statement node that shall be inserted.
        :param n: the position at which the statement node shall be inserted.
        """
        self.child.insert(n if n is not None else len(self.child), statement_node)

    def prepend_statement(self, statement_node: ASTNode):
        """
        Inserts the given statement at the beginning of the statement block.
        :param statement_node: the statement node that shall be inserted at the beginning of the statement block.
        """
        self.child.insert(0, statement_node)

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
        self.child.append(var_expr)
        self.child.append(value_expr)

    @property
    def variable_expression(self):
        return self.child[0]

    @property
    def value_expression(self):
        return self.child[1]

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

    def __init__(self, func_name, trunk, parameter_list=None, *args, **kwargs):
        super(FunctionNode, self).__init__(*args, **kwargs)
        self.environment = None
        self.child.append(func_name)
        self.child.append(trunk)
        self.child.append(parameter_list)

    @property
    def function_name(self):
        return self.child[0]

    @property
    def trunk(self):
        return self.child[1]

    @property
    def parameter_list(self):
        return self.child[2]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        self.environment = environment
        total_parameters = dict()
        default_parameters = dict()
        for index, parameter_node in enumerate(self.parameter_list):
            total_parameters[index] = parameter_node.parameter_name
            if parameter_node.default_expression is not None:
                default_parameters[parameter_node.parameter_name] = parameter_node.default_expression
        environment.insert_variable(name=self.function_name, type=Function,
                                    value=Function(name=self.function_name, ast_node=self.trunk,
                                                   environment=environment,
                                                   total_parameters=total_parameters,
                                                   default_parameters=default_parameters))
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self) -> str:
        return '(%s def ..%s.. %s { %s }' % (
            self.__class__.__name__, self.function_name, repr(self.parameter_list) if self.parameter_list else '',
            repr(self.trunk))


class ParameterNode(ASTNode):
    def __init__(self, parameter_name, default_expression, *args, **kwargs):
        super(ParameterNode, self).__init__(*args, **kwargs)
        self.child.append(default_expression)
        self.parameter_name = parameter_name

    @property
    def default_expression(self):
        return self.child[0]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        pass

    def __repr__(self) -> str:
        return '(%s %s %s)' % (
            self.__class__.__name__, self.parameter_name,
            '= %s' % repr(self.default_expression) if self.default_expression else '')


class ParameterListNode(ASTNode):
    """ This class represents a list of parameters. """

    def __init__(self, parameter_node=None, *args, **kwargs):
        super(ParameterListNode, self).__init__(*args, **kwargs)
        self.parameter_names = dict()
        self.default_expression_parameters = dict()
        self.index_count = 0
        if parameter_node is not None:
            self.insert_parameter(parameter_node)

    @property
    def total_parameters_count(self):
        return self.index_count

    @property
    def fixed_parameters_count(self):
        return self.total_parameters_count - len(self.default_expression_parameters)

    def insert_parameter(self, parameter_node: ParameterNode):
        """
        Inserts the given parameter into the parameter list node.
        :param parameter_node: the parameter node that shall be appended to the parameter list.
        """
        self.parameter_names[self.index_count] = parameter_node.parameter_name
        self.child.append(parameter_node)
        if parameter_node.default_expression is not None:
            self.default_expression_parameters[parameter_node.parameter_name] = parameter_node.default_expression
        self.index_count += 1

    def execute(self, environment: Environment, program_stack: ProgramStack):
        pass


class FunctionArgumentNode(ASTNode):
    """ This class represents a function argument."""

    def __init__(self, arg_expr, name=None, *args, **kwargs):
        super(FunctionArgumentNode, self).__init__(*args, **kwargs)
        self.arg_expr = arg_expr
        self.name = name

    def execute(self, environment: Environment, program_stack: ProgramStack):
        # The default expressions are executed by the called function node.
        pass

    def __repr__(self):
        return '(%s %s %s)' % (
            self.__class__.__name__, '%s =' % self.name if self.name is not None else '', self.arg_expr)


class FunctionArgumentListNode(ASTNode):
    """ This class represents a list of function arguments."""

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
            self.fixed_arguments.append(function_argument.arg_expr)
        else:
            self.named_arguments[function_argument.name] = function_argument.arg_expr
        self.child.append(function_argument)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        pass


class TestListNode(ASTNode):
    def __init__(self, test_node=None, *args, **kwargs):
        super(TestListNode, self).__init__(*args, **kwargs)
        if test_node is not None:
            self.append_test_node(test_node)

    def append_test_node(self, test_node: ASTNode):
        """
        Appends the given test node at the end of the list.
        :param test_node: the test node that shall be appended at the end of the test list.
        """
        self.child.append(test_node)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        value_list = list()
        for test_node in self.children:
            value_list.append(test_node.execute(environment, program_stack).value)
        return ASTExecutionResult(ASTExecutionResultType.value_, value_list)


class FunctionCallNode(ASTNode):
    """ This class represents the call of a function with the given amount of arguments. """

    def __init__(self, left_side_expression, argument_list=None, *args, **kwargs):
        super(FunctionCallNode, self).__init__(*args, **kwargs)
        self.child.append(left_side_expression)
        self.child.append(argument_list)

    @property
    def left_side_expression(self):
        return self.child[0]

    @property
    def argument_list(self):
        return self.child[1]

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
    """ This class represents pass node that does nothing. It is is used to indicate a statement. """

    def __init__(self, *args, **kwargs):
        super(PassNode, self).__init__(*args, **kwargs)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.void_, None)


class PrintNode(ASTNode):
    def __init__(self, print_expression, *args, **kwargs):
        super(PrintNode, self).__init__(*args, **kwargs)
        self.child.append(print_expression)

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

    @abstractmethod
    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.continue_, None)


class BreakNode(FlowControlNode):
    def __init__(self, *args, **kwargs):
        super(BreakNode, self).__init__(*args, **kwargs)

    @abstractmethod
    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.break_, None)


class IfOperationNode(FlowControlNode):
    def __init__(self, test, true_branch, else_branch=None, *args, **kwargs):
        super(IfOperationNode, self).__init__(*args, **kwargs)
        self.child.append(test)
        self.child.append(true_branch)
        self.child.append(else_branch)

    @property
    def test(self) -> ASTNode:
        return self.child[0]

    @property
    def true_branch(self):
        return self.child[1]

    @property
    def else_branch(self) -> ASTNode:
        return self.child[2]

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
        self.child.append(test)
        self.child.append(trunk)
        self.child.append(else_branch)

    @property
    def test(self):
        return self.child[0]

    @property
    def trunk(self):
        return self.child[1]

    @property
    def else_branch(self):
        return self.child[2]

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
        self.child.append(iterable_node)
        self.child.append(trunk)
        self.child.append(else_branch)
        self.variable_name = variable_name

    @property
    def iterable_node(self):
        return self.child[0]

    @property
    def trunk(self):
        return self.child[1]

    @property
    def else_branch(self):
        return self.child[2]

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
        self.child.append(return_expr)

    @property
    def return_expr(self):
        return self.child[0]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        rtrn_value = self.return_expr.execute(environment,
                                              program_stack).value if self.return_expr is not None else None
        return ASTExecutionResult(ASTExecutionResultType.return_, rtrn_value)

    def __repr__(self) -> str:
        return '(%s return %s)' % (self.__class__.__name__, '' if self.return_expr is None else repr(self.return_expr))


class OperationNode(ASTNode):
    def __init__(self, op_name, magic_method, *args, **kwargs):
        super(OperationNode, self).__init__(*args, **kwargs)
        self.op_name = op_name
        self.magic_method = magic_method

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
        self.child.append(left)
        self.child.append(right)

    @property
    def left_operand(self):
        return self.child[0]

    @property
    def right_operand(self):
        return self.child[1]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        left_operand = self.left_operand.execute(environment, program_stack).value
        right_operand = self.right_operand.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  getattr(left_operand, self.magic_method)(right_operand))

    def __repr__(self) -> str:
        return '(%s %s %s %s)' % (
            self.__class__.__name__, repr(self.child[0]) if self.child[0] is not None else '(None)', self.op_name,
            repr(self.child[1]) if self.child[1] is not None else '(None)')


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
        self.child.append(node)

    @property
    def kid(self):
        return self.child[0]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        kid_value = self.kid.execute(environment, program_stack).value
        return ASTExecutionResult(ASTExecutionResultType.value_, getattr(kid_value, self.magic_method)())

    def __repr__(self) -> str:
        return '(%s %s %s)' % (
            self.__class__.__name__, self.op_name, repr(self.child[0]) if self.child[0] is not None else '(None)')


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
        """ Initializes a constant node with the given value. """
        super(ConstantNode, self).__init__(*args, **kwargs)
        self.value = value

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.value_, self.value)

    def __repr__(self) -> str:
        return self.value.__repr__()


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
        self.child.append(atom_node)

    def execute(self, environment: Environment, program_stack: ProgramStack):
        value_list = list()
        for atom_node in self.children:
            value_list.append(atom_node.execute(environment).value)
        return ASTExecutionResult(ASTExecutionResultType.value_, value_list)


class GraphNode(ASTNode):
    def __init__(self, graph_construction_node=None, *args, **kwargs):
        super(GraphNode, self).__init__(*args, **kwargs)
        self.child.append(graph_construction_node)

    @property
    def construction_node(self):
        return self.child[0]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  graph(self.construction_node.execute(
                                      environment).value) if self.construction_node is not None else graph())


class ListNode(ASTNode):
    def __init__(self, list_construction_node, *args, **kwargs):
        super(ListNode, self).__init__(*args, **kwargs)
        self.child.append(list_construction_node)

    @property
    def list_construction_node(self):
        return self.child[0]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        if self.list_construction_node is None:
            return ASTExecutionResult(ASTExecutionResultType.value_, list())
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  self.list_construction_node.execute(environment, program_stack).value)


class SubscriptNode(ASTNode):
    """ An abstract syntax tree node that creates a iterable list from the given node in the given range. """

    def __init__(self, node, lower=None, upper=None, *args, **kwargs):
        super(SubscriptNode, self).__init__(*args, **kwargs)
        self.child.append(node)
        self.child.append(lower)
        self.child.append(upper)

    @property
    def kid(self):
        return self.child[0]

    @property
    def lower_bound(self):
        return self.child[1]

    @property
    def upper_bound(self):
        return self.child[2]

    def execute(self, environment: Environment, program_stack: ProgramStack):
        iter_object = self.child[0].execute(environment).value
        lower = self.lower_bound.execute(environment, program_stack).value if self.lower_bound is not None else 0
        upper = self.upper_bound.execute(environment, program_stack).value if self.upper_bound is not None else None
        if upper is not None and upper < 0: upper = len(iter_object) - upper
        result = list()
        for index, x in enumerate(iter_object):
            if index < lower:
                continue
            elif upper is not None and index > upper:
                continue
            else:
                result.append(x)
        return ASTExecutionResult(ASTExecutionResultType.value_, result)
