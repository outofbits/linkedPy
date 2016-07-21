# COPYRIGHT (c) 2016 Kevin Haller <kevin.haller@outofbits.com>

import logging

from os.path import join, exists, isfile
from collections import OrderedDict, deque
from env import ProgramContainer
from exception import ByteCodeConstantNotFound, ByteCodeFileNotFound, ByteCodeCorruptedError
from ast import (ASTNode, NoneNode, FalseNode, TrueNode, NumberNode, StringNode, byte_ast_dispatch)

logger = logging.getLogger(__name__)

code_base_header = bytes([0xC0, 0xDE, 0xBA, 0x5E])
code_program_chapter = bytes([0x04])
code_hash_length = 28

ast_dispatch_map = {
    NoneNode.identifier: NoneNode,
    FalseNode.identifier: FalseNode,
    TrueNode.identifier: TrueNode,
    NumberNode.identifier: NumberNode,
    StringNode.identifier: StringNode,
}


class ConstantPool(object):
    constant_identifier_start = 0xC0
    identifier = bytes([constant_identifier_start])
    identifier_length = 1

    def __init__(self):
        self._pool_counter = 0
        self._pool_dict = OrderedDict()
        self._constant_list = list()

    def add(self, constant) -> int:
        """
        Adds a constant to the constant pool and returns the index that has been assigned to the value. If there is
        already a equivalent entry in the constant pool, only the index is returned. The index will be returned as
        byte code that can be placed in the byte code as representative for the given constant.
        :param constant: constant that shall be added to this constant pool.
        :return: the index of the given constant that has been inserted into the constant pool.
        """
        if constant in self._pool_dict:
            return self._constant_byte_code(self._pool_dict[constant])
        else:
            index = self._pool_counter
            self._pool_dict[constant] = index
            self._pool_counter += 1
            self._constant_list.append(constant)
            return self._constant_byte_code(index)

    def get(self, byte_code):
        """
        Decodes the given byte code for a constant that represents the index of the constant in this constant pool. If
        this entry exists, the corresponding value is returned, otherwise a ByteCodeConstantNotFound will be thrown.
        :param byte_code: the byte code of the index of the constant that shall be returned.
        :return: the value of the constant that is represented by the given byte code.
        """
        index = int.from_bytes(byte_code, byteorder='little')
        if index >= self._pool_counter:
            print('>> %s' % index)
            raise ByteCodeConstantNotFound('%s was not found in the constant pool.' % byte_code)
        return self._constant_list[index]

    def constant_index_size(self, constant_header):
        """
        Returns the size of the constant according to the header of the constant.
        :param constant_header: the header of the constant of which the size shall be returned.
        :return: the size of the constant index according to the constant header.
        """
        return int.from_bytes(constant_header, byteorder='little') - self.constant_identifier_start

    def cache(self) -> bytearray:
        """
        Transforms the constant pool into a byte array so that it can be reconstructed to an equivalent constant pool.
        :return: the byte array representing the constant pool.
        """
        pool_bytes = bytearray(self.identifier)
        for elem in self._pool_dict:
            pool_bytes.extend(elem.to_bytes())
        return pool_bytes

    def construct_from_cache(self, fd):
        """
        Constructs the constant pool from the byte array that the file descriptor is pointing to.
        :param fd: the file descriptor that points to the byte array representing a constant pool.
        :return: the constant pool reconstructed from the given byte array.
        """
        next_b = fd.read(self.identifier_length)
        while next_b in [NoneNode.identifier, TrueNode.identifier, FalseNode.identifier, NumberNode.identifier,
                         StringNode.identifier]:
            self.add(ast_dispatch_map[next_b].from_bytes(fd))
            next_b = fd.read(self.identifier_length)
        fd.seek(-self.identifier_length, 1)

    @classmethod
    def _constant_byte_code(cls, index) -> bytearray:
        """
        Gets the constant bytes code for the constant with the given index.
        :param index: the index of the constant of which the bytes code shall be returned.
        :return: constant bytes code for the constant with the given index.
        """
        off = 1
        constant_byte_list = deque()
        while True:
            constant_byte_list.appendleft(index & 0xFF)
            index >>= 8
            if index == 0:
                break
            off += 1
        if off > 9:
            raise ValueError('The limit for constants is 2^256.')
        constant_byte_list.appendleft(cls.constant_identifier_start + off)
        return bytearray(constant_byte_list)

    def __repr__(self):
        return '{Constant-Pool: %s}' % repr(self._pool_dict)


def generate_tree_based_intermediate_code(root: ASTNode, program_container: ProgramContainer):
    """
    Generates a tree-based intermediate code for the program represented by the given AST node.
    :param root: the root node of the abstract syntax tree that represents the AST node.
    :param program_container: the container of the program that shall be generated.
    """
    constant_pool = ConstantPool()
    program_trunk_byte = root.cache(constant_pool)
    # Concatenates the byte representation of the program.
    program_byte = bytearray(code_base_header)
    program_byte.extend(program_container.hash_digest)
    program_byte.extend(constant_pool.cache())
    program_byte.extend(code_program_chapter)
    program_byte.extend(program_trunk_byte)
    # Write out the bytecode.
    if program_container.program_name is None:
        raise ValueError('The program container has too less information for caching the program.')
    with open(join(program_container.program_dir, program_container.program_name + '.lpyc'), 'wb') as out_fd:
        out_fd.write(program_byte)


def ast_tree_of_intermediate_code(program_container: ProgramContainer) -> ASTNode:
    """
    This method checks for the cache file of the given program. If the cache file exists, the corresponding abstract
    syntax tree will be restored.
    :param program_container: the container of the program of which the cached abstract syntax tree shall be read in.
    :return:  the abstract syntax tree stored in the cache file of the given program, None if there is no such file, or
    the program changed.
    """
    # Check for the cache file.
    if program_container.program_name is None:
        raise ValueError('The program container has too less information for caching the program.')
    program_cache_file = join(program_container.program_dir, program_container.program_name + '.lpyc')
    if not exists(program_cache_file) or not isfile(program_cache_file):
        raise ByteCodeFileNotFound('Cache-File %s was not found !' % program_cache_file)
    # Decode the cache file
    with open(program_cache_file, 'rb') as cache_fd:
        header = cache_fd.read(len(code_base_header))
        # Checks the header of the cached file.
        if code_base_header != header:
            raise ByteCodeCorruptedError('The header of the cached file of the program \'%s\' is corrupted (%s).' % (
                program_container.origin, header))
        del header
        # Compares the hash value of the given program and the hash value contained in the cached file.
        cached_hash_digest = cache_fd.read(code_hash_length)
        if cached_hash_digest != program_container.hash_digest:
            raise ByteCodeCorruptedError(
                'The hash value of the cached file \'%s\' differs from the given program (%s, %s)' % (
                    program_container.origin, cached_hash_digest, program_container.hash_digest))
        del cached_hash_digest
        next_b = cache_fd.read(len(ConstantPool.identifier))
        # Load in the constant pool at the begin of the cached file.
        constant_pool = ConstantPool()
        if ConstantPool.identifier == next_b:
            constant_pool.construct_from_cache(cache_fd)
            logger.debug('Read in constant pool from file \'%s\': %s' % (program_container.origin, constant_pool))
        else:
            raise ByteCodeCorruptedError('There is no constant pool for %s.' % program_container.origin)
        # Restores the abstract syntax tree.
        next_b = cache_fd.read(len(code_program_chapter))
        if next_b != code_program_chapter:
            raise ByteCodeCorruptedError(
                'The program body of the cached file \'%s\' is corrupted.' % program_container.origin)
        return byte_ast_dispatch[cache_fd.read(ASTNode.identifier_length)].construct_from_cache(cache_fd, constant_pool,
                                                                                                program_container)
