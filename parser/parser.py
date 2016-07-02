# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

import ply.yacc as yacc
from .tokenizer import Tokenizer
from .exception import SyntaxError, ParserError, ParsersError
from ast.ast import *


class Parser:
    """ This class represents a parser that reads in a string, breaks it into linkedPy tokens, checks the syntax and
        transform the read-in program into a Abstract Syntax Tree (AST).
    """

    tokens = Tokenizer.tokens
    start = 'program'

    def __init__(self, program_origin=None):
        self.parser = yacc.yacc(module=self, optimize=0, debug=True)
        self.program_origin = program_origin
        self.syntax_error_list = list()

    precedence = (
        ('left', 'or'),
        ('left', 'and'),
        ('right', 'not'),
        ('left', 'EQEQUAL', 'NOTEQUAL', '<', '>', 'LESSEQUAL', 'GREATEREQUAL', 'is', 'in'),
        ('left', '|'),
        ('left', '^'),
        ('left', '&'),
        ('left', 'LSHIFT', 'RSHIFT'),
        ('left', '+', '-'),
        ('left', '*', '/', 'FLOORDIVIDE', '%'),
        ('right', 'UPLUS', 'UMINUS', '~'),
        ('left', 'POWER'),
    )

    binary_op_magic_methods_dic = {
        '+': '__add__',
        '-': '__sub__',
        '*': '__mul__',
        '/': '__truediv__',
        '%': '__mod__',
        '|': '__or__',
        '^': '__xor__',
        '&': '__and__',
        '<<': '__lshift__',
        '>>': '__rshift__',
        '**': '__pow__',
    }

    assign_binop_magic_methods_dic = {
        '+=': '__add__',
        '-=': '__sub__',
        '*=': '__mul__',
        '/=': '__truediv__',
        '%=': '__mod__',
        '|=': '__or__',
        '^=': '__xor__',
        '&=': '__and__',
        '<<=': '__lshift__',
        '>>=': '__rshift__',
        '**=': '__pow__',
    }

    cmp_magic_methods_dic = {
        '==': '__eq__',
        '!=': '__ne__',
        '<': '__lt__',
        '<=': '__le__',
        '>': '__gt__',
        '>=': '__ge__',
    }

    unary_op_magic_methods_dic = {
        '+': '__pos__',
        '-': '__neg__',
        '~': '__invert__'
    }

    def p_program(self, p):
        """
        program : file_input
        """
        p[0] = p[1] if p != 'empty' else None

    def p_file_input(self, p):
        """
        file_input : file_body
        """
        p[0] = p[1]

    def p_file_body(self, p):
        """
        file_body : file_header
                  | file_body statement
                  | file_body NEWLINE
        """
        if len(p) == 2:
            p[0] = StatementsBlockNode([p[1]])
        else:
            if p[2] != '\n':
                p[1].append_statement(p[2])
            p[0] = p[1]

    def p_file_header(self, p):
        """
        file_header : empty
                    | file_header NEWLINE
                    | file_header prefix_statement
        """
        if len(p) == 2:
            p[0] = PrefixNodeList()
        else:
            if p[2] != '\n':
                p[1].append_prefix(p[2])
            p[0] = p[1]

    # ------------------------------------------------------------------------
    #                                Prefix
    # ------------------------------------------------------------------------

    def p_prefix_statement(self, p):
        """
         prefix_statement : '@' base ':' '<' iri '>' '.'
                          | '@' prefix NAME ':' '<' iri '>' '.'
        """
        p[0] = PrefixNode(name='base', iri=p[5]) if len(p) == 8 else PrefixNode(name=p[3], iri=p[6])

    # ------------------------------------------------------------------------
    #                          Function definition
    # ------------------------------------------------------------------------

    def p_function_definition(self, p):
        """
        function_definition : def NAME function_parameters ':' suite
        """
        p[0] = FunctionNode(p[2], p[5], p[3])

    def p_function_parameters(self, p):
        """
        function_parameters : '(' ')'
                            | '(' parameter_list  ')'
        """
        if len(p) == 3:
            p[0] = ParameterListNode()
        else:
            p[0] = p[2]

    def p_parameter_list(self, p):
        """
        parameter_list : parameter
                       | parameter_list ',' parameter
        """
        if len(p) == 2:
            p[0] = ParameterListNode(p[1])
        else:
            p[1].insert_parameter(p[3])
            p[0] = p[1]

    def p_parameter(self, p):
        """
        parameter : NAME parameter_default
                  | NAME ':' test parameter_default
        """
        if len(p) == 3:
            p[0] = ParameterNode(parameter_name=p[1], default_expression=p[2])
        else:
            p[0] = ParameterNode(parameter_name=p[1], default_expression=p[4])

    def p_parameter_test(self, p):
        """
        parameter_default : empty
                          | '=' test
        """
        p[0] = p[2] if len(p) == 3 else None

    # ------------------------------------------------------------------------
    #                          Statements
    # ------------------------------------------------------------------------

    def p_suite(self, p):
        """
        suite : simple_statement
              | NEWLINE INDENT statement_list DEDENT
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[3]

    def p_statement_list(self, p):
        """
        statement_list : statement
                       | statement_list statement
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = StatementsBlockNode.merge(p[1], p[2])

    def p_statement(self, p):
        """
        statement : simple_statement NEWLINE
                  | simple_statement ';' NEWLINE
                  | compound_statement
        """
        p[0] = p[1]

    def p_simple_statement(self, p):
        """
        simple_statement : single_statement
                         | simple_statement ';' single_statement
        """
        if len(p) == 2:
            p[0] = StatementsBlockNode([p[1]])
        else:
            p[0] = StatementsBlockNode.merge(p[1], p[2])

    def p_single_statement(self, p):
        """
        single_statement : expr
                         | assign_statement
                         | flow_statement
                         | pass_statement
                         | print_statement
        """
        p[0] = p[1]

    def p_compound_statement(self, p):
        """
        compound_statement : function_definition
                           | if_statement
                           | while_statement
                           | for_statement
        """
        p[0] = StatementsBlockNode([p[1]])

    def p_if_statement(self, p):
        """
        if_statement : if test ':' suite elif_statement
        """
        p[0] = IfOperationNode(p[2], p[4], p[5])

    def p_elif_statement(self, p):
        """
        elif_statement : empty
                       | else ':' suite
                       | elif test ':' suite elif_statement
        """
        if len(p) == 2:
            p[0] = None
        elif len(p) == 4:
            p[0] = p[3]
        else:
            p[0] = IfOperationNode(p[2], p[4], p[5])

    def p_while_statement(self, p):
        """
        while_statement : while test ':' suite
                        | while test ':' suite else ':' suite
        """
        if len(p) == 5:
            p[0] = WhileOperationNode(p[2], p[4])
        else:
            p[0] = WhileOperationNode(p[2], p[4], p[7])

    def p_for_statement(self, p):
        """
        for_statement : for NAME in test ':' suite
        """
        p[0] = ForOperationNode(variable_name=p[2], iterable_node=p[4], trunk=p[6])

    def p_assign_statement(self, p):
        """
        assign_statement : test '=' test
                         | test PLUSEQUAL test
                         | test MINUSEQUAL test
                         | test TIMESEQUAL test
                         | test DIVIDEQUAL test
                         | test MODULOEQUAL test
                         | test LANDEQUAL test
                         | test LOREQUAL test
                         | test XOREQUAL test
                         | test RSHIFTEQUAL test
                         | test LSHIFTEQUAL test
                         | test POWEREQUAL test
                         | test FLOORDIVIDEQUAL test
        """
        if p[2] == '=':
            p[0] = VariableAssignmentNode(p[1], p[3])
        else:
            p[0] = VariableAssignmentNode(p[1], BinOperationNode(op_name=p[2],
                                                                 magic_method=self.assign_binop_magic_methods_dic[p[2]],
                                                                 left=p[1],
                                                                 right=p[3]))

    def p_print_statement(self, p):
        """
        print_statement : print '(' test ')'
        """
        p[0] = PrintNode(p[3])

    def p_pass_statement(self, p):
        """
        pass_statement : pass
        """
        p[0] = PassNode()

    def p_flow_statement(self, p):
        """
        flow_statement : break_statement
                       | continue_statement
                       | return_statement
        """
        p[0] = p[1]

    def p_break_statement(self, p):
        """
        break_statement : break
        """
        pass

    def p_continue_statement(self, p):
        """
        continue_statement : continue
        """
        pass

    def p_return_statement(self, p):
        """
        return_statement : return
                         | return test
        """
        p[0] = ReturnNode() if len(p) == 2 else ReturnNode(return_expr=p[2])

    # -----------------------------------------------------------------------
    #                            Expressions & Tests
    # -----------------------------------------------------------------------

    def p_test(self, p):
        """
        test : bool_test
             | bool_test if bool_test else test
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = None

    def p_boolean_test(self, p):
        """
        bool_test : not_test
                  | bool_test or bool_test
                  | bool_test and bool_test
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ANDOperationNode(p[1], p[3]) if p[2] == 'and' else OROperationNode(p[1], p[3])

    def p_test_not(self, p):
        """
        not_test : expr
                 | not not_test
        """
        if p[1] == 'not':
            p[0] = NotOperationNode(p[2])
        else:
            p[0] = p[1]

    def p_test_list(self, p):
        """
        test_list : test
                  | test_list ',' test
        """
        if len(p) == 2:
            p[0] = TestListNode(p[1])
        else:
            p[1].append_test_node(p[3])
            p[0] = p[1]

    def p_expr(self, p):
        """
        expr : atom_expr
             | expr_unary
             | arithmetic_expr
             | comparison_expr
             | bits_expr
        """
        p[0] = p[1]

    def p_expr_unary(self, p):
        """
        expr_unary : '(' expr ')'
                   | '+' expr_unary %prec UPLUS
                   | '-' expr_unary %prec UMINUS
                   | '~' expr_unary
        """
        if p[1] in self.unary_op_magic_methods_dic.keys():
            p[0] = UnaryOperatorNode(op_name=p[1], magic_method=self.unary_op_magic_methods_dic[p[1]], node=p[2])
        else:
            p[0] = p[2]

    def p_arithmetic_expr(self, p):
        """
        arithmetic_expr : expr '+' expr
                        | expr '-' expr
                        | expr '*' expr
                        | expr '/' expr
                        | expr FLOORDIVIDE expr
                        | expr '%' expr
                        | expr POWER expr
        """
        p[0] = BinOperationNode(op_name=p[2], magic_method=self.binary_op_magic_methods_dic[p[2]], left=p[1],
                                right=p[3])

    def p_bits_expr(self, p):
        """
        bits_expr : expr '|' expr
                  | expr '^' expr
                  | expr '&' expr
                  | expr LSHIFT expr
                  | expr RSHIFT expr
        """
        p[0] = BinOperationNode(op_name=p[2], magic_method=self.binary_op_magic_methods_dic[p[2]], left=p[1],
                                right=p[3])

    def p_comparison_expr(self, p):
        """
        comparison_expr : expr '<' expr
                        | expr '>' expr
                        | expr EQEQUAL expr
                        | expr NOTEQUAL expr
                        | expr LESSEQUAL expr
                        | expr GREATEREQUAL expr
                        | expr in expr
                        | expr not in expr
                        | expr is expr
                        | expr is not expr
        """
        if p[2] in self.cmp_magic_methods_dic.keys():
            p[0] = ComparisonOperationNode(op_name=p[2], magic_method=self.cmp_magic_methods_dic[p[2]], left=p[1],
                                           right=p[3])
        else:
            p[0] = None

    # -------------------------------------------------------------------
    #                      Atoms
    # -------------------------------------------------------------------

    def p_atom_expr(self, p):
        """
        atom_expr : atom
                  | '+' atom_expr
                  | '-' atom_expr
                  | '~' atom_expr
        """
        if p[1] in self.unary_op_magic_methods_dic.keys():
            p[0] = UnaryOperatorNode(op_name=p[1], magic_method=self.unary_op_magic_methods_dic[p[1]], node=p[2])
        else:
            p[0] = p[1]

    def p_atom(self, p):
        """
        atom : '(' atom ')'
             | variable
             | resource
             | triple
             | func_call
             | subscript_call
             | const_num
             | const_bool
             | str
             | list_atom
             | graph_atom
             | none
        """
        p[0] = p[1]

    def p_none(self, p):
        """
        none : None
        """
        p[0] = ConstantNode(None)

    def p_str(self, p):
        """
        str : STRING
        """
        p[0] = ConstantNode(p[1])

    def p_const_number(self, p):
        """
        const_num :  NUMBER
        """
        p[0] = ConstantNode(p[1])

    def p_const_bool(self, p):
        """
        const_bool : True
                   | False
        """
        p[0] = ConstantNode(p[1] == 'True')

    def p_resource(self, p):
        """
        resource : '<' '>'
                 | '<' iri '>'
                 | '<' ':' iri '>'
                 | '<' NAME ':' iri '>'
        """
        if len(p) == 3:
            p[0] = ResourceNode(iri='')
        elif len(p) == 4:
            p[0] = ResourceNode(iri=p[2])
        elif len(p) == 5:
            p[0] = ResourceNode(iri=p[3], prefix_name='base')
        else:
            p[0] = ResourceNode(iri=p[4], prefix_name=p[2])

    def p_iri(self, p):
        """
        iri : IRI
            | NAME
        """
        p[0] = p[1]

    def p_triple(self, p):
        """
        triple : '(' test ',' test ',' test ')'
        """
        p[0] = TripleNode(p[2], p[4], p[6])

    def p_variable(self, p):
        """
        variable : NAME
        """
        p[0] = VariableNode(p[1])

    def p_graph_atom(self, p):
        """
        graph_atom : '{' '}'
                   | '{' graph_construction '}'
        """
        if len(p) == 3:
            p[0] = GraphNode()
        else:
            p[0] = GraphNode(p[2])

    def p_graph_construction(self, p):
        """
        graph_construction : atom
                           | graph_construction ',' atom
        """
        if len(p) == 2:
            p[0] = AtomListNode(p[1])
        else:
            p[1].append_atom_node(p[3])
            p[0] = p[1]

    def p_list_atom(self, p):
        """
        list_atom : '[' ']'
                  | '[' list_construction ']'
        """
        p[0] = ListNode(None) if len(p) == 3 else ListNode(p[2])

    def p_list_construction(self, p):
        """
        list_construction : test_list
        """
        p[0] = p[1]

    def p_func_call(self, p):
        """
        func_call : atom '(' ')'
                  | atom '(' argument_list ')'
        """
        p[0] = FunctionCallNode(p[1]) if len(p) == 4 else FunctionCallNode(p[1], p[3])

    def p_subscript_call(self, p):
        """
        subscript_call : atom '[' test ']'
                       | atom '[' test ':' ']'
                       | atom '[' ':' test ']'
                       | atom '[' test ':' test ']'
        """
        if len(p) == 5:
            p[0] = BinOperationNode(op_name='subscript', magic_method='__getitem__', left=p[1], right=p[3])
        elif len(p) == 6:
            if p[4] == ':':
                p[0] = SubscriptNode(p[1], lower=p[3])
            else:
                p[0] = SubscriptNode(p[1], upper=p[5])
        else:
            p[0] = SubscriptNode(p[1], lower=p[3], upper=p[5])

    def p_argument_list(self, p):
        """
        argument_list : argument
                      | argument_list ',' argument
        """
        if len(p) == 2:
            p[0] = FunctionArgumentListNode(p[1])
        else:
            p[1].insert_argument(p[3])
            p[0] = p[1]

    def p_argument(self, p):
        """
        argument : test
                 | NAME '=' test
        """
        if len(p) == 2:
            p[0] = FunctionArgumentNode(arg_expr=p[1])
        else:
            p[0] = FunctionArgumentNode(name=p[1], arg_expr=p[3])

    def p_empty(self, p):
        """empty :"""
        pass

    def p_error(self, p):
        if p:
            self.syntax_error_list.append(
                SyntaxError(err_msg='invalid syntax.', lexdata=p.lexer.lexdata, lineno=p.lineno, lexpos=p.lexpos))
            self.parser.errok()
        else:
            self.syntax_error_list.append(ParserError('syntax error at the end of the file.'))

    def parse(self, program_input: str):
        """
        Parses the given string.
        :param prog_input: the string of the program that shall be parsed.
        :return:
        """
        tokenizer = Tokenizer()
        response = self.parser.parse(program_input, lexer=tokenizer.indentation_lexer())
        if self.syntax_error_list:
            raise ParsersError(error_msg='The program could not be parsed.', parser_errors=self.syntax_error_list)
        return response
