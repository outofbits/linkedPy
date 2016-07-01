# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

from .env import Environment, FunctionDescription
from datatypes.linkedtypes import resource, triple, graph
from abc import abstractmethod
from enum import Enum
from collections import namedtuple


class ASTExecutionResultType(Enum):
    void_ = 1
    value_ = 2
    return_ = 3
    continue_ = 4
    break_ = 5


ASTExecutionResult = namedtuple('ASTExecutionResult', ['type', 'value'])


class ASTNode(object):
    """ This class represents a node of an abstract syntax tree. """

    def __init__(self, line: str = '', lineno: int = None, linepos: int = None):
        self.child = []
        self.line = line
        self.lineno = lineno
        self.linepos = linepos

    @property
    def children(self):
        return self.child

    @abstractmethod
    def execute(self, environment: Environment) -> ASTExecutionResult:
        """
        Executes this ASTNode and returns the corresponding result of the execution.
        :param environment: the environment in which context this ASTNode shall be executed.
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


class StatementsBlockNode(ASTNode):
    """ This class represents an abstract syntax tree that hold a sequence of child nodes that shall be executed in the
        given order.
    """

    def __init__(self, statements=None):
        super(StatementsBlockNode, self).__init__()
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

    def execute(self, environment: Environment):
        for statement in self.children:
            result = statement.execute(environment)
            if result.type == ASTExecutionResultType.void_ or result.type == ASTExecutionResultType.value_:
                continue
            else:
                return result
        return ASTExecutionResult(ASTExecutionResultType.void_, None)


class VariableAssignmentNode(ASTNode):
    def __init__(self, var_expr, value_expr):
        super(VariableAssignmentNode, self).__init__()
        self.child.append(var_expr)
        self.child.append(value_expr)

    @property
    def variable_expression(self):
        return self.child[0]

    @property
    def value_expression(self):
        return self.child[1]

    def execute(self, environment: Environment) -> ASTExecutionResult:
        var_response = self.variable_expression.prepare(environment)
        if var_response.type == ASTExecutionResultType.value_:
            value_response = self.value_expression.execute(environment)
            if value_response.type in [ASTExecutionResultType.void_, ASTExecutionResultType.value_]:
                if var_response.value is None:
                    environment.insert_variable(self.variable_expression.name, value=value_response.value)
                else:
                    var_response.value.change_value(value_response.value)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self) -> str:
        return '(%s %s = %s)' % (
            self.__class__.__name__, repr(self.variable_expression), repr(self.value_expression))


class FunctionNode(ASTNode):
    """ This class represents a function definition with a non-empty trunk and none, one or more arguments. """

    def __init__(self, func_name, trunk, parameter_list=None):
        super(FunctionNode, self).__init__()
        self.environment = None
        self.func_name = func_name
        self.child.append(trunk)
        self.child.append(parameter_list)

    @property
    def trunk(self):
        return self.child[0]

    @property
    def parameter_list(self):
        return self.child[1]

    def _local_environment(self, parent_environment: Environment) -> Environment:
        local_env = Environment(parent_environment)
        default_params = self.parameter_list.default_expression_parameters
        for param_name in default_params.keys():
            param_expr_result = default_params[param_name].execute(parent_environment).value
            local_env.insert_variable(param_name, value=param_expr_result)
        return local_env

    def call(self, *fixed_args, **named_args):
        loc_env = self._local_environment(self.environment)
        for index, value in enumerate(fixed_args):
            loc_env.insert_variable(name=self.parameter_list.parameter_names[index], value=value)
        for arg_name in named_args.keys():
            loc_env.insert_variable(name=arg_name, value=named_args[arg_name])
        return ASTExecutionResult(ASTExecutionResultType.value_, self.trunk.execute(loc_env).value);

    def execute(self, environment: Environment):
        self.environment = environment
        environment.insert_function(self.func_name, self, self.parameter_list.total_parameters_count,
                                    self.parameter_list.fixed_parameters_count)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self) -> str:
        return '(%s def ..%s.. %s { %s }' % (
            self.__class__.__name__, self.func_name, repr(self.parameter_list) if self.parameter_list else '',
            repr(self.trunk))


class ParameterNode(ASTNode):
    def __init__(self, parameter_name, default_expression):
        super(ParameterNode, self).__init__()
        self.child.append(default_expression)
        self.parameter_name = parameter_name

    @property
    def default_expression(self):
        return self.child[0]

    def execute(self, environment: Environment) -> ASTExecutionResult:
        pass

    def __repr__(self) -> str:
        return '(%s %s %s)' % (
            self.__class__.__name__, self.parameter_name,
            '= %s' % repr(self.default_expression) if self.default_expression else '')


class ParameterListNode(ASTNode):
    """ This class represents a list of parameters. """

    def __init__(self, parameter_node=None):
        super(ParameterListNode, self).__init__()
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

    def execute(self, environment: Environment) -> ASTExecutionResult:
        pass


class FunctionArgumentNode(ASTNode):
    """ This class represents a function argument."""

    def __init__(self, arg_expr, name=None):
        super(FunctionArgumentNode, self).__init__()
        self.arg_expr = arg_expr
        self.name = name

    def execute(self, environment: Environment) -> ASTExecutionResult:
        # The default expressions are executed by the called function node.
        pass

    def __repr__(self):
        return '(%s %s %s)' % (
            self.__class__.__name__, '%s =' % self.name if self.name is not None else '', self.arg_expr)


class FunctionArgumentListNode(ASTNode):
    """ This class represents a list of function arguments."""

    def __init__(self, func_arg: FunctionArgumentNode):
        super(FunctionArgumentListNode, self).__init__()
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

    def execute(self, environment: Environment) -> ASTExecutionResult:
        pass


class TestListNode(ASTNode):
    def __init__(self, test_node=None):
        super(TestListNode, self).__init__()
        if test_node is not None:
            self.append_test_node(test_node)

    def append_test_node(self, test_node: ASTNode):
        """
        Appends the given test node at the end of the list.
        :param test_node: the test node that shall be appended at the end of the test list.
        """
        self.child.append(test_node)

    def execute(self, environment: Environment) -> ASTExecutionResult:
        value_list = list()
        for test_node in self.children:
            value_list.append(test_node.execute(environment).value)
        return ASTExecutionResult(ASTExecutionResultType.value_, value_list)


class FunctionCallNode(ASTNode):
    """ This class represents the call of a function with the given amount of arguments. """

    def __init__(self, var_expr, argument_list=None):
        super(FunctionCallNode, self).__init__()
        self.var_expr = var_expr
        self.child.append(argument_list)

    @property
    def argument_list(self):
        return self.child[0]

    def execute(self, environment: Environment):
        var_description = self.var_expr.prepare(environment).value
        if not isinstance(var_description.value, FunctionDescription):
            raise ValueError('%s is not a function.' % var_description.name)
        function_description = environment.get_function(var_description.value.name)
        if function_description is None:
            raise ValueError('Function %s was not declared before !' % var_description.name)
        fixed_arguments = list()
        named_arguments = dict()
        if self.argument_list:
            if self.argument_list.total_arguments_count > function_description.total_parameters_count:
                raise ValueError(
                    'More arguments (%d) given than taken by the method %s. At least %d arguments must be given and at most %d !' % (
                        self.argument_list.total_arguments_count, function_description.name,
                        function_description.fixed_parameters_count, function_description.total_parameters_count))
            if self.argument_list.total_arguments_count < function_description.fixed_parameters_count:
                raise ValueError(
                    'Less arguments (%d) given than taken by the method %s. At least %d arguments must be given and at most %d !' % (
                        self.argument_list.total_arguments_count, function_description.name,
                        function_description.fixed_parameters_count, function_description.total_parameters_count))
            # Compute arguments
            for fixed_arg in self.argument_list.fixed_arguments:
                fixed_arguments.append(fixed_arg.execute(environment).value)
            for arg_name in self.argument_list.named_arguments:
                named_arguments[arg_name] = self.argument_list.named_arguments[arg_name].execute(environment).value
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  function_description.ast_node.call(*fixed_arguments, **named_arguments).value)

    def __repr__(self) -> str:
        return '(%s %s %s)' % (
            self.__class__.__name__, repr(self.var_expr), repr(self.argument_list) if self.argument_list else '()')


class PassNode(ASTNode):
    """ This class represents pass node that does nothing. It is is used to indicate a statement. """

    def __init__(self):
        super(PassNode, self).__init__()

    def execute(self, environment: Environment) -> ASTExecutionResult:
        return ASTExecutionResult(ASTExecutionResultType.void_, None)


class PrintNode(ASTNode):
    def __init__(self, print_expression):
        super(PrintNode, self).__init__()
        self.child.append(print_expression)

    def execute(self, environment: Environment):
        print(self.children[0].execute(environment).value)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)


class FlowControlNode(ASTNode):
    """ This class represents a node that maybe redirects the program flow. """

    def __init__(self):
        super(FlowControlNode, self).__init__()

    @abstractmethod
    def execute(self, environment: Environment) -> ASTExecutionResult:
        super(FlowControlNode, self).execute(environment)


class IfOperationNode(FlowControlNode):
    def __init__(self, test, true_branch, else_branch=None):
        super(IfOperationNode, self).__init__()
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

    def execute(self, environment: Environment):
        test_value = self.test.execute(environment).value
        if test_value:
            return self.true_branch.execute(environment)
        elif self.else_branch is not None:
            return self.else_branch.execute(environment)
        else:
            return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self) -> str:
        return '(%s if %s then %s %s' % (self.__class__.__name__, repr(self.test), repr(self.true_branch),
                                         ' else %s' % repr(self.else_branch) if self.else_branch is not None else '')


class WhileOperationNode(FlowControlNode):
    def __init__(self, test, trunk, else_branch=None):
        super(WhileOperationNode, self).__init__()
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

    def execute(self, environment: Environment) -> ASTExecutionResult:
        while True:
            test_cond = self.test.execute(environment).value
            if test_cond:
                trunk_response = self.trunk.execute(environment)
                if trunk_response.type == ASTExecutionResultType.return_:
                    return trunk_response
                elif trunk_response.type == ASTExecutionResultType.continue_:
                    continue
                elif trunk_response.type == ASTExecutionResultType.break_:
                    return ASTExecutionResult(ASTExecutionResultType.void_, None)
            else:
                break
        if self.else_branch is not None:
            else_response = self.else_branch.execute(environment)
            if else_response.type == ASTExecutionResultType.return_:
                return else_response
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self) -> str:
        return '(%s while %s do %s %s' % (self.__class__.__name__, repr(self.test), repr(self.trunk),
                                          ' else %s' % repr(self.else_branch) if self.else_branch is not None else '')


class ForOperationNode(ASTNode):
    def __init__(self, variable_name: str, iterable_node: ASTNode, trunk: ASTNode):
        super(ForOperationNode, self).__init__()
        self.child.append(iterable_node)
        self.child.append(trunk)
        self.variable_name = variable_name

    @property
    def iterable_node(self):
        return self.child[0]

    @property
    def trunk(self):
        return self.child[1]

    def execute(self, environment: Environment):
        local_env = Environment(environment)
        iterable_obj = iter(self.iterable_node.execute(environment).value)
        for entry in iterable_obj:
            local_env.insert_variable(self.variable_name, value=entry)
            trunk_response = self.trunk.execute(local_env)
            if trunk_response.type == ASTExecutionResultType.return_:
                return trunk_response
            elif trunk_response.type == ASTExecutionResultType.continue_:
                continue
            elif trunk_response.type == ASTExecutionResultType.break_:
                return ASTExecutionResult(ASTExecutionResultType.void_, None)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self):
        return '(%s for %s in %s do %s)' % (self.__class__.__name__, self.variable_name, self.iterable_node, self.trunk)


class ReturnNode(FlowControlNode):
    def __init__(self, return_expr=None):
        super(ReturnNode, self).__init__()
        self.child.append(return_expr)

    @property
    def return_expr(self):
        return self.child[0]

    def execute(self, environment: Environment):
        rtrn_value = self.return_expr.execute(environment).value if self.return_expr is not None else None
        return ASTExecutionResult(ASTExecutionResultType.return_, rtrn_value)

    def __repr__(self) -> str:
        return '(%s return %s)' % (self.__class__.__name__, '' if self.return_expr is None else repr(self.return_expr))


class OperationNode(ASTNode):
    def __init__(self, op_name, magic_method):
        super(OperationNode, self).__init__()
        self.op_name = op_name
        self.magic_method = magic_method

    @abstractmethod
    def execute(self, environment: Environment) -> ASTExecutionResult:
        super(OperationNode, self).execute(environment)


class BinOperationNode(OperationNode):
    def __init__(self, op_name, magic_method, left=None, right=None):
        """
        Initializes a new binary operator node with the given left and right child.
        :param op_name: the name of the operation.
        :param magic_method: the magic methods (right, left) for this operation.
        :param left: the left child of the binary operator.
        :param right: the right child of the binary operator.
        """
        super(BinOperationNode, self).__init__(op_name, magic_method)
        self.child.append(left)
        self.child.append(right)

    @property
    def left_operand(self):
        return self.child[0]

    @property
    def right_operand(self):
        return self.child[1]

    def execute(self, environment: Environment):
        left_operand = self.left_operand.execute(environment).value
        right_operand = self.right_operand.execute(environment).value
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  getattr(left_operand, self.magic_method)(right_operand))

    def __repr__(self) -> str:
        return '(%s %s %s %s)' % (
            self.__class__.__name__, repr(self.child[0]) if self.child[0] is not None else '(None)', self.op_name,
            repr(self.child[1]) if self.child[1] is not None else '(None)')


class ComparisonOperationNode(BinOperationNode):
    def __init__(self, op_name, magic_method, left=None, right=None):
        super(ComparisonOperationNode, self).__init__(op_name, magic_method, left, right)


class BooleanBinOperationNode(BinOperationNode):
    def __init__(self, op_name, left=None, right=None):
        super(BooleanBinOperationNode, self).__init__(op_name, None, left, right)


class ANDOperationNode(BooleanBinOperationNode):
    def __init__(self, left=None, right=None):
        super(BooleanBinOperationNode, self).__init__('and', None, left, right)

    def execute(self, environment: Environment):
        left_operand = self.left_operand.execute(environment).value
        right_operand = self.right_operand.execute(environment).value
        return ASTExecutionResult(ASTExecutionResultType.value_, left_operand and right_operand)


class OROperationNode(BooleanBinOperationNode):
    def __init__(self, left=None, right=None):
        super(BooleanBinOperationNode, self).__init__('or', None, left, right)

    def execute(self, environment: Environment):
        left_operand = self.left_operand.execute(environment).value
        right_operand = self.right_operand.execute(environment).value
        return ASTExecutionResult(ASTExecutionResultType.value_, left_operand or right_operand)


class UnaryOperatorNode(OperationNode):
    def __init__(self, op_name, magic_method, node=None):
        """
        Initializes a new unary operator node with the given node.
        :param op_name: the name of the unary operation.
        :param magic_method: the magic method for this operation
        :param node: the child of this operation node.
        """
        super(UnaryOperatorNode, self).__init__(op_name, magic_method)
        self.child.append(node)

    @property
    def kid(self):
        return self.child[0]

    def execute(self, environment: Environment):
        kid_value = self.kid.execute(environment).value
        return ASTExecutionResult(ASTExecutionResultType.value_, getattr(kid_value, self.magic_method)())

    def __repr__(self) -> str:
        return '(%s %s %s)' % (
            self.__class__.__name__, self.op_name, repr(self.child[0]) if self.child[0] is not None else '(None)')


class UnaryBooleanOperationNode(UnaryOperatorNode):
    def __init__(self, op_name, node=None):
        super(UnaryBooleanOperationNode, self).__init__(op_name, None, node)


class NotOperationNode(UnaryBooleanOperationNode):
    def __init__(self, node=None):
        super(NotOperationNode, self).__init__('not', node)

    def execute(self, environment: Environment):
        kid_value = self.kid.execute(environment).value
        return ASTExecutionResult(ASTExecutionResultType.value_, not kid_value)


class ConstantNode(ASTNode):
    def __init__(self, value):
        """ Initializes a constant node with the given value. """
        super(ConstantNode, self).__init__()
        self.value = value

    def execute(self, environment: Environment) -> ASTExecutionResult:
        return ASTExecutionResult(ASTExecutionResultType.value_, self.value)

    def __repr__(self) -> str:
        return self.value.__repr__()


class VariableNode(ASTNode):
    def __init__(self, name):
        super(VariableNode, self).__init__()
        self.name = name

    def prepare(self, environment: Environment):
        var_description = environment.get_variable(self.name)
        return ASTExecutionResult(ASTExecutionResultType.value_, var_description)

    def execute(self, environment: Environment) -> ASTExecutionResult:
        var_description = self.prepare(environment)
        if var_description is None:
            raise ValueError('Variable %s was not declared before.' % self.name)
        return ASTExecutionResult(ASTExecutionResultType.value_, (var_description.value).value)

    def __repr__(self) -> str:
        return self.name.__repr__()


class PrefixNode(ASTNode):
    def __init__(self, name, iri=''):
        super(PrefixNode, self).__init__()
        self.name = name
        self.iri = iri

    def execute(self, environment: Environment):
        if self.name is not None:
            environment.insert_prefix(name=self.name, iri=self.iri)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)

    def __repr__(self):
        return '(%s %s : %s)' % (self.__class__.__name__, self.name, self.iri)


class PrefixNodeList(ASTNode):
    def __init__(self, prefix_node=None):
        super(PrefixNodeList, self).__init__()
        if prefix_node is not None:
            self.append_prefix(prefix_node)

    def append_prefix(self, prefix_node: PrefixNode):
        """
        Appends the given prefix node to this prefix node list.
        :param prefix_node: the prefix node that shall be appended to this prefix node list.
        """
        self.child.append(prefix_node)

    def execute(self, environment: Environment):
        for prefix_node in self.children:
            prefix_node.execute(environment)
        return ASTExecutionResult(ASTExecutionResultType.void_, None)


class ResourceNode(ASTNode):
    def __init__(self, iri, prefix_name=None):
        super(ResourceNode, self).__init__()
        self.prefix_name = prefix_name
        self.iri = iri

    def execute(self, environment: Environment) -> ASTExecutionResult:
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
    def __init__(self, subject, predicate, object):
        super(TripleNode, self).__init__()
        self.subject = subject
        self.predicate = predicate
        self.object = object

    def execute(self, environment: Environment) -> ASTExecutionResult:
        subj = self.subject.execute(environment).value
        pred = self.predicate.execute(environment).value
        obj = self.object.execute(environment).value
        return ASTExecutionResult(ASTExecutionResultType.value_, triple((subj, pred, obj)))

    def __repr__(self):
        return '(%s %s %s)' % (self.subject, self.predicate, self.object)


class AtomListNode(ASTNode):
    def __init__(self, atom_node=None):
        super(AtomListNode, self).__init__()
        if atom_node is not None:
            self.append_atom_node(atom_node)

    def append_atom_node(self, atom_node: ASTNode):
        """
        Appends the given test node at the end of the list.
        :param atom_node: the test node that shall be appended at the end of the test list.
        """
        self.child.append(atom_node)

    def execute(self, environment: Environment) -> ASTExecutionResult:
        value_list = list()
        for atom_node in self.children:
            value_list.append(atom_node.execute(environment).value)
        return ASTExecutionResult(ASTExecutionResultType.value_, value_list)


class GraphNode(ASTNode):
    def __init__(self, graph_construction_node=None):
        super(GraphNode, self).__init__()
        self.child.append(graph_construction_node)

    @property
    def construction_node(self):
        return self.child[0]

    def execute(self, environment: Environment):
        return ASTExecutionResult(ASTExecutionResultType.value_,
                                  graph(self.construction_node.execute(
                                      environment).value) if self.construction_node is not None else graph())


class ListNode(ASTNode):
    def __init__(self, list_construction_node):
        super(ListNode, self).__init__()
        self.child.append(list_construction_node)

    @property
    def list_construction_node(self):
        return self.child[0]

    def execute(self, environment: Environment) -> ASTExecutionResult:
        if self.list_construction_node is None:
            return ASTExecutionResult(ASTExecutionResultType.value_, list())
        return ASTExecutionResult(ASTExecutionResultType.value_, self.list_construction_node.execute(environment).value)


class SubscriptNode(ASTNode):
    """ An abstract syntax tree node that creates a iterable list from the given node in the given range. """

    def __init__(self, node, lower=None, upper=None):
        super(SubscriptNode, self).__init__()
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

    def execute(self, environment: Environment) -> ASTExecutionResult:
        iter_object = self.child[0].execute(environment).value
        lower = self.lower_bound.execute(environment).value if self.lower_bound is not None else 0
        upper = self.upper_bound.execute(environment).value if self.upper_bound is not None else None
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
