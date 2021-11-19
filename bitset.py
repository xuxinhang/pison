# Implement BitSet util function.


def create_bitset(size: int):
    return bytearray((size + 7) // 8)


def get_bit(bitset: bytearray, idx: int):
    i, j = idx // 8, idx % 8
    return (bitset[i] >> j) & 1


def set_bit(bitset: bytearray, idx: int):
    i, j = idx // 8, idx % 8
    bitset[i] |= 1 << j


def clr_bit(bitset: bytearray, idx: int):
    i, j = idx // 8, idx % 8
    bitset[i] &= ~(1 << j)


def or_bitset(bitset: bytearray, source: bytearray):
    for i in range(len(source)):
        if source[i] != 0:  # @HACK: it seems Python's bit operator is slow.
        # if ((~bitset[i]) & source[i]) != 0:
            bitset[i] |= source[i]


def iterate_bitset(bitset: bytearray, zero=False):
    idx = 0
    for bank in bitset:
        for _ in range(8):
            if bank & 1:
                yield idx
            bank = bank >> 1
            idx += 1




