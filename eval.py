# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

import argparse
import logging
import sys

from env import GlobalEnvironment
from env import ProgramContainer
from exception import ExecutionError
from parser.exception import ParserErrors
from parser.parser import Parser

logger = logging.getLogger(__name__)


def evaluate(program_String, program_origin='unknown'):
    """
    Evaluates the given program string.
    :param program_String: the string of the program that shall be evaluated.
    :param program_origin: the origin of the program code like the file path.
    """
    program_container = ProgramContainer(origin=program_origin, program_string=program_String)
    try:
        parsed_program = Parser.parse(program_container)
        logger.debug('Parsed AST for %s: %s' % (program_origin, parsed_program))
        global_env = GlobalEnvironment(name='__main__', file_path=program_origin)
        parsed_program.execute(global_env, None)
        logger.debug('Environment after execution: %s' % global_env)
    except (ParserErrors, ExecutionError) as p:
        print(p.message(), file=sys.stderr)


def evaluate_program_file(program_path):
    """
    Evaluates the program that is contained in the file with the given path.
    :param program_path: the path to the linked python program that shall be executed.
    """
    program = ''
    with open(program_path, 'r') as fp:
        for line in fp:
            program += line
    evaluate(program, program_origin=program_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpreter for LinkedPython.')
    parser.add_argument('--path', '-p', type=str, help='a path to the linkedPy program that shall be interpreted.')
    parser.add_argument('--debug', '-d', help='indicates that debug output shall be printed out.', action='store_true')
    args = parser.parse_args()
    # Optional debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    # Path argument
    if not args.path:
         pass # evaluate_command_line(), not implemented.
    else:
        evaluate_program_file(args.path)
