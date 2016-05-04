# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

import grammar.keyword as keyword

import re
import ply.lex as lex
from .exception import SyntaxError, IndentError
from ply.lex import TOKEN, LexToken

__author__ = 'Kevin Haller'


def group(*choices:[str]) -> str: return r'(' + '|'.join(choices) + ')'


class Tokenizer:
    """
    This class represents a tokenizer, which reads in a string and breaks it into
    linkedPy tokens. linkedPy inherits token from Python3 and adds new ones for
    handling linked data. Tokens are tuples of the form (token type, token value).
    """

    def __init__(self):
        self._last_indent_space = 0  # Space counter for indentation tokens.
        self.lexer = lex.lex(object=self, reflags=re.MULTILINE)

    comment = r'\#[^\r\n]*'
    newline = r'\n'
    space = r'[ \t]+'
    # Names
    name = r'[a-zA-Z_]\w*'
    # Number formats
    binnumber = r'0[bB][01]*'
    hexnumber = r'0[xX][\da-fA-F]*[lL]?'
    octnumber = r'0[oO]?[0-7]*[lL]?'
    decnumber = r'[1-9]\d*[lL]?'
    number = group(binnumber, hexnumber, octnumber, decnumber)
    # String formats
    apostrophe_string = r'\'[^\'\\]*(\\.[^\'\\]*)*\''
    quote_string = r'"[^"\\]*(\\.[^"\\]*)*"'
    string = group(apostrophe_string, quote_string)

    literals = (':', '.', ',', ';',
                '(', ')', '[', ']', '{', '}',
                '+', '-', '*', '/','%',
                '&','|','~', '^',
                '<','>',
                '@'
                )

    tokens = ('INDENT',
              'DEDENT',
              'NAME',
              'NUMBER',
              'STRING',
              'NEWLINE',
              'PLUSEQUAL',
              'MINUSEQUAL',
              'TIMESEQUAL',
              'DIVIDEQUAL',
              'INTDIVIDE',
              'INTDIVIDEQUAL',
              'MODULOEQUAL',
              'LOREQUAL',
              'LANDEQUAL',
              'XOREQUAL',
              'POWER',
              'POWEREQUAL',
              'GREATEREQUAL',
              'LESSEQUAL',
              'EQEQUAL',
              'NOTEQUAL',
              'ATEQUAL',
              'LSHIFT',
              'LSHIFTEQUAL',
              'RSHIFT',
              'RSHIFTEQUAL',
              'RIGHTARROW',
              'ELLIPSIS',
              ) + keyword.kwlist

    t_ATEQUAL = r'@='
    t_RIGHTARROW = r'->'
    t_ELLIPSIS = r'\.\.\.'
    t_POWER = r'\*\*'
    t_INTDIVIDE = r'\\\\'
    t_NOTEQUAL = r'!='
    t_EQEQUAL = r'=='
    t_LESSEQUAL = r'<='
    t_GREATEREQUAL = r'>='
    t_MINUSEQUAL = r'-='
    t_PLUSEQUAL = r'\+='
    t_TIMESEQUAL = r'\*='
    t_POWEREQUAL = r'\*\*='
    t_DIVIDEQUAL = r'\\='
    t_INTDIVIDEQUAL = r'\\\\='
    t_MODULOEQUAL = r'%='
    t_XOREQUAL = r'\^='
    t_LANDEQUAL = r'&='
    t_LOREQUAL = r'\|='
    t_LSHIFT = r'<<'
    t_LSHIFTEQUAL = r'<<='
    t_RSHIFT = r'>>'
    t_RSHIFTEQUAL = r'>>='

    @TOKEN(space)
    def t_SPACE(self, t: LexToken) -> LexToken:
        space_counter = 0
        for c in reversed(t.lexer.lexdata[:t.lexer.lexpos]):
            if c == '\n':
                break
            elif c not in ['\t', ' ']:
                return None
            else:
                space_counter += 1 if c == ' ' else 4
        t.value = space_counter if space_counter >= self._last_indent_space else self._last_indent_space - space_counter
        t.type = 'INDENT' if space_counter >= self._last_indent_space else 'DEDENT'
        self._last_indent_space = space_counter
        return t

    @TOKEN(comment)
    def t_COMMENT(self, t: LexToken) -> None:
        pass

    @TOKEN(name)
    def t_NAME(self, t: LexToken) -> LexToken:
        t.type = 'NAME' if not keyword.isKeyword(t.value) else t.value
        return t

    @TOKEN(number)
    def t_NUMBER(self, t: LexToken) -> LexToken:
        t.value = int(t.value)
        return t

    @TOKEN(string)
    def t_STRING(self, t: LexToken) -> LexToken:
        return t

    @TOKEN(newline)
    def t_NEWLINE(self, t: LexToken) -> LexToken:
        t.lexer.lineno += 1
        return t

    def t_error(self, t:LexToken) -> LexToken:
        raise SyntaxError(t.value, t.lineno, t.lexpos, err_msg='invalid syntax')

    def tokenize(self, string: str) -> [()]:
        """ Analyses the given string and breaks it into

        :param string: the string, which shall be broken into linkedPy token.
        :return: a iterator over linkedPy tokens from beginning to the end of the file.
        """
        self._last_indent_space = 0     # Space counter for indentation tokens.
        self.lexer.input(string)
        for tok in iter(lex.token, None):
            print('%s, %s' % (repr(tok.type), repr(tok.value)))
