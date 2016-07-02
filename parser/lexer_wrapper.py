# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

import copy
from collections import deque
from ply.lex import LexToken, Lexer
from .exception import IndentationError


class LexerIndentationWrapper(object):
    """ The PLY does not support a push parser interface, where multiple token can be returned at once (e.g. for a
        newline). This class represents a wrapper for inserting indentation tokens and detecting syntactical errors
        like diverging space number for the same indentation level.
    """

    def __init__(self, wrapped_lexer):
        """
        Initializes a new lexer wrapper for inserting indentation tokens.
        :param wrapped_lexer: the lexer that shall be used by the wrapper.
        """
        self.wrapped_lexer = wrapped_lexer
        self._wrap_lex_ind_queue = [0]  # Internal queue to store the space number of the indentation levels.
        self._wrap_lex_token_queue = deque()  #

    def _count_leading_whitespaces(self, token: LexToken) -> int:
        """
        Replaces all new tab characters with 4 white space characters and counts the leading white space characters
        for the current position of the lexer
        :return: the number of leading white spaces for the current position of lexer (tab characters are replaced with
        whitespaces)
        """
        if self.wrapped_lexer.lexpos == self.wrapped_lexer.lexlen:
            return 0
        else:
            response = 0
            for c in token.lexer.lexdata[token.lexer.lexpos:]:
                if c == ' ':
                    response += 1
                elif c == '\t':
                    response += 4
                else:
                    break
            return response

    def _new_token(self, type=None, value=None, lexpos=None, lineno=None) -> LexToken:
        """
        Creates a new lexer token with the given properties.
        :return: a new lexer token with the given properties.
        """
        token = LexToken()
        token.type = type
        token.value = value
        token.lexpos = lexpos
        token.lineno = lineno

    def _change_token(self, token: LexToken, new_type=None, new_value=None):
        """
        Copies the given token and changes it according to the given parameters new_type and new_value.

        :param token: the token that shall be copied and changed.
        :param new_type: the new type of the copied token.
        :param new_value: the new value of the copied token.
        :return: the new copied token.
        """
        if token is None: return None
        copied_token = copy.copy(token)
        copied_token.type = new_type
        copied_token.value = new_value
        return copied_token

    def input(self, string: str):
        """
        Delegates the given string to the input() - method of the lexer.
        :param string: the program that shall be parsed.
        """
        self.wrapped_lexer.input(string)

    def token(self):
        """
        Wrapper method around the token() - method of lexer. This wrapper method inserts indentation and dedentation
        token at the right place.
        :return: the next token.
        """
        if self._wrap_lex_token_queue:
            return self._wrap_lex_token_queue.pop()
        token = self.wrapped_lexer.token()
        if token is None:
            self._wrap_lex_token_queue.append(None)
            while self._wrap_lex_ind_queue[-1] != 0:
                self._wrap_lex_ind_queue.pop()
                self._wrap_lex_token_queue.append(self._new_token('DEDENT', self._wrap_lex_ind_queue[-1]))
            return self._wrap_lex_token_queue.pop()
        elif token.type == 'NEWLINE':
            indent_c = self._count_leading_whitespaces(token)
            if indent_c > self._wrap_lex_ind_queue[-1]:
                self._wrap_lex_ind_queue.append(indent_c)
                self._wrap_lex_token_queue.append(self._change_token(token, 'INDENT', indent_c))
            else:
                while indent_c < self._wrap_lex_ind_queue[-1]:
                    self._wrap_lex_ind_queue.pop()
                    self._wrap_lex_token_queue.append(self._change_token(token, 'DEDENT', self._wrap_lex_ind_queue[-1]))
                if indent_c != self._wrap_lex_ind_queue[-1]:
                    raise IndentationError(err_msg="Indentation error.", lineno=token.lineno, lexpos=token.lexpos,
                                           lexdata=token.lexer.lexdata)
        return token

    def __setattr__(self, key, value):
        if key.startswith('lex'):
            setattr(self.wrapped_lexer, key, value)
        else:
            super(LexerIndentationWrapper, self).__setattr__(key, value)

    def __getattr__(self, name):
        if name.startswith('lex'):
            return getattr(self.wrapped_lexer, name)
        else:
            raise AttributeError('Class %s has no attribute %s.' % (self.__class__.__name__, name))
