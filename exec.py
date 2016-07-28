# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

import sys
import argparse
import logging

from os.path import abspath, basename, dirname
from env import GlobalEnvironment, ProgramContainer
from exception import ExecutionError, ParserErrors, IntermediateCodeError
from parser.parser import Parser
from codegeneration import generate_tree_based_intermediate_code, ast_tree_of_intermediate_code

logger = logging.getLogger(__name__)


def execute(program_container: ProgramContainer):
    """
    Executes the given program.
    :param program_container: the program container containing the program that shall be executed.
    """
    try:
        try:
            parsed_program = ast_tree_of_intermediate_code(program_container)
        except IntermediateCodeError as b:
            logger.error(b.message())
            parsed_program = Parser.parse(program_container)
            generate_tree_based_intermediate_code(parsed_program, program_container)
        logger.debug('Parsed AST for %s: %s' % (program_container.origin, parsed_program))
        global_env = GlobalEnvironment(name='__main__', file_path=program_container.origin)
        parsed_program.execute(global_env, None)
        logger.debug('Environment after execution: %s' % global_env)
    except (ParserErrors, ExecutionError) as p:
        print(p.message(), file=sys.stderr)


def execute_program_file(program_path):
    """
    Evaluates the program that is contained in the file with the given path.
    :param program_path: the path to the linked python program that shall be executed.
    """
    with open(program_path, 'r') as fp:
        program = fp.read()
    # Prepare the program container
    program_path = abspath(program_path)
    program_dir = dirname(program_path)
    program_basename = basename(program_path)
    execute(ProgramContainer(program_string=program, origin=program_path, program_dir=program_dir,
                             program_basename=program_basename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpreter for LinkedPython.')
    parser.add_argument('path', type=str, help='a path to the linkedPy program that shall be interpreted.')
    parser.add_argument('--debug', '-d', help='indicates that debug output shall be printed out.', action='store_true')
    args = parser.parse_args()
    # Optional debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    # Path argument
    if args.path:
        logging.basicConfig(level=logging.FATAL)
        execute_program_file(args.path)
