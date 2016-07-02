# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

import grammar.keyword as keyword

import re
import ply.lex as lex
from .lexer_wrapper import LexerIndentationWrapper
from .exception import SyntaxError
from ply.lex import TOKEN, LexToken

__author__ = 'Kevin Haller'


def group(*choices: [str]) -> str: return r'(' + '|'.join(choices) + ')'


def optional(expression: str) -> str: return r'(%s)?' % expression


def concat(*expressions: [str]) -> str: return r''.join(expressions)


class Tokenizer:
    """
    This class represents a tokenizer that reads in a string and breaks it into
    linkedPy tokens. linkedPy inherits token from Python3 and adds new ones for
    handling linked data. Tokens are tuples of the form (token type, token value).
    """

    def __init__(self, program_origin: str = None):
        self.brace_level = 0
        self.program_origin = program_origin
        self.name_regex = re.compile(concat(r'^', self.name, r'$'), flags=re.UNICODE)
        self.lexer = lex.lex(module=self, optimize=1, debug=False, reflags=re.MULTILINE | re.UNICODE)

    states = (
        ('inbraces', 'exclusive'),
    )

    comment = r'^\#[^\r\n]*'
    newline = r'\n+'
    # Names
    name = r'[a-zA-Z_]\w*'
    # IRI
    iri_scheme = r'([^\W\d_<>\s]([^\W\s<>]|[+.-])*)'
    iri_username = r'([^:@/<>\s]+(:[^:@/<>\s]+)?@)'
    iri_segment = r'([^\W\d_<>\s](([^\W<>\s])*[^\W<>\s])?)'
    iri_domain = concat(iri_segment, r'(\.', iri_segment, r')*')
    iri_host = concat(iri_domain)
    iri_port = r'(:[\d]{1,5})'
    iri_path = r'((/([^?;\W\#<>\s]|[+.-])*)*)'
    iri_query = r'(\?([^\W\#;<>\s]|[+.-])*)'
    iri_anchor = r'(\#([^\W<>\s]|[+.-])*)'
    iri = concat(optional(concat(iri_scheme, r'://')), iri_host, optional(iri_port),
                 optional(iri_path), optional(iri_query), optional(iri_anchor))
    # Number formats
    binnumber = r'0[bB][01]*'
    hexnumber = r'0[xX][\da-fA-F]*[lL]?'
    octnumber = r'0[oO]?[0-7]*[lL]?'
    decnumber = r'[1-9]\d*[lL]?'
    number = group(binnumber, hexnumber, octnumber, decnumber)
    # String formats
    apostrophe_string = r'\'([^\'\\]*(\\.[^\'\\]*)*)\''
    quote_string = r'"([^"\\]*(\\.[^"\\]*)*)"'
    apostrophe3_string = r'\'\'\'([^\'\\]*((\\.|\'(?!\'\'))[^\'\\]*)*)\'\'\''
    quote3_string = r'"""([^"\\]*((\\.|"(?!""))[^"\\]*)*)"""'
    string = group(apostrophe3_string, quote3_string, apostrophe_string, quote_string)
    string_group_enc = (8, 10, 2, 5)
    # Braces
    braces_open = r'[\(\[{]'
    braces_close = r'[\)\]}]'

    literals = (':', '.', ',', ';',
                '(', ')', '[', ']', '{', '}',
                '+', '-', '*', '/', '%',
                '&', '|', '~', '^',
                '<', '>', '=',
                '@'
                )

    tokens = ('INDENT',
              'DEDENT',
              'IRI',
              'NAME',
              'NUMBER',
              'STRING',
              'NEWLINE',
              'PLUSEQUAL',
              'MINUSEQUAL',
              'TIMESEQUAL',
              'DIVIDEQUAL',
              'FLOORDIVIDE',
              'FLOORDIVIDEQUAL',
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
              'LSHIFT',
              'LSHIFTEQUAL',
              'RSHIFT',
              'RSHIFTEQUAL',
              ) + keyword.kwlist

    t_ignore = ' \t'
    t_inbraces_ignore = ' \t\n'
    t_ANY_ATEQUAL = r'@='
    t_ANY_POWER = r'\*\*'
    t_ANY_FLOORDIVIDE = r'//'
    t_ANY_NOTEQUAL = r'!='
    t_ANY_EQEQUAL = r'=='
    t_ANY_LESSEQUAL = r'<='
    t_ANY_GREATEREQUAL = r'>='
    t_ANY_MINUSEQUAL = r'-='
    t_ANY_PLUSEQUAL = r'\+='
    t_ANY_TIMESEQUAL = r'\*='
    t_ANY_POWEREQUAL = r'\*\*='
    t_ANY_DIVIDEQUAL = r'/='
    t_ANY_FLOORDIVIDEQUAL = r'//='
    t_ANY_MODULOEQUAL = r'%='
    t_ANY_XOREQUAL = r'\^='
    t_ANY_LANDEQUAL = r'&='
    t_ANY_LOREQUAL = r'\|='
    t_ANY_LSHIFT = r'<<'
    t_ANY_LSHIFTEQUAL = r'<<='
    t_ANY_RSHIFT = r'>>'
    t_ANY_RSHIFTEQUAL = r'>>='

    def _is_name(self, string: str) -> bool:
        return self.name_regex.match(string) is not None

    @TOKEN(comment)
    def t_ANY_COMMENT(self, t: LexToken) -> None:
        pass

    @TOKEN(braces_open)
    def t_ANY_BRACES_OPEN(self, t: LexToken):
        t.type = t.value
        if self.brace_level == 0:
            t.lexer.begin('inbraces')
        self.brace_level += 1
        return t

    @TOKEN(braces_close)
    def t_ANY_BRACES_CLOSE(self, t: LexToken):
        t.type = t.value
        self.brace_level -= 1
        if self.brace_level == 0:
            t.lexer.begin('INITIAL')
        return t

    @TOKEN(string)
    def t_ANY_STRING(self, t: LexToken) -> LexToken:
        for g in self.string_group_enc:
            str_value = t.lexer.lexmatch.group(g)
            if str_value:
                t.value = str_value
                break
        return t

    @TOKEN(iri)
    def t_ANY_IRI(self, t: LexToken) -> LexToken:
        if self._is_name(t.value):
            t.type = 'NAME' if not keyword.isKeyword(t.value) else t.value
        return t

    @TOKEN(number)
    def t_ANY_NUMBER(self, t: LexToken) -> LexToken:
        t.value = int(t.value)
        return t

    @TOKEN(newline)
    def t_NEWLINE(self, t: LexToken) -> LexToken:
        t.lexer.lineno += len(t.value)
        t.value = t.value[0]
        return t

    @TOKEN(newline)
    def t_INBRACES_NEWLINE(self, t: LexToken) -> LexToken:
        t.lexer.lineno += len(t.value)
        pass

    def t_ANY_error(self, t: LexToken) -> LexToken:
        raise SyntaxError(lexdata=self.lexer.lexdata, lineno=t.lineno, lexpos=t.lexpos, err_msg='invalid syntax',
                          origin=self.program_origin)

    def indentation_lexer(self) -> LexerIndentationWrapper:
        """
        Gets a lexer that includes the indentation token as well as checks for proper indention.
        :return: a lexer that includes the indentation token as well as checks for proper indention.
        """
        return LexerIndentationWrapper(self.lexer)

    def tokenize(self, string: str) -> [()]:
        """ Analyses the given string and breaks it into

        :param string: the string, which shall be broken into linkedPy token.
        :return: a iterator over linkedPy tokens from beginning to the end of the file.
        """
        self._last_indent_space = 0  # Space counter for indentation tokens.
        self.lexer.input(string)
        for tok in iter(self.indentation_lexer().token, None):
            print('%s, %s' % (repr(tok.type), repr(tok.value)))
