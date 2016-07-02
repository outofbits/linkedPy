# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

import argparse
import sys
import logging

from parser.parser import Parser
from parser.exception import ParsersError
from ast.env import Environment


def evaluate_command_line():
    """
    Evaluates the linked py instructions entered on the command line.
    """
    # TODO: Implement


def evaluate(program, program_origin='unknown'):
    """
    Evaluates the given program string.
    :param program: the string of the program that shall be evaluated.
    :param program_origin: the origin of the program code like the file path.
    """
    program_parser = Parser(program_origin)
    try:
        parsed_program = program_parser.parse(program)
        print('Execute ... ')
        global_env = Environment()
        parsed_program.execute(global_env)
        logging.debug(global_env)
    except ParsersError as p:
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
    args = parser.parse_args()
    if not args.path:
        evaluate_command_line()
    else:
        evaluate_program_file(args.path)
