# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

import argparse
import logging
import sys
from os.path import abspath, basename, dirname

from env import GlobalEnvironment
from env import ProgramContainer
from exception import ExecutionError
from exception import ParserErrors
from parser.parser import Parser

logger = logging.getLogger(__name__)


def execute(program_container: ProgramContainer):
    """
    Executes the given program.
    :param program_container: the program container containing the program that shall be executed.
    """
    try:
        parsed_program = Parser.parse(program_container)
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
    parser.add_argument('--path', '-p', type=str, help='a path to the linkedPy program that shall be interpreted.')
    parser.add_argument('--debug', '-d', help='indicates that debug output shall be printed out.', action='store_true')
    args = parser.parse_args()
    # Optional debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    # Path argument
    if not args.path:
        pass  # evaluate_command_line(), not implemented.
    else:
        execute_program_file(args.path)
