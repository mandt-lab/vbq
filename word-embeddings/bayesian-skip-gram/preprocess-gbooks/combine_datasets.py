'''TODO: document
'''

from models.lib import dataset
from scipy import sparse

import argparse
import struct
import sys
import gzip
import numpy as np


def add_cli_args(parser):
    parser.add_argument('input', metavar='IN_PATH_TEMPLATE', help='''
        Path to a binary file containing the co-occurrence counts for an individual time step.
        Must contain exactly one integer placeholder (such as "%%s"), which will be replaced
        by integers from --from to --to.''')
    parser.add_argument('output', metavar='OUT_PATH', help='''
        Path to the output file. Must not already exist.''')

    parser.add_argument('--first', metavar='N', type=int, required=True, help='''
        Id of the first time step (inclusive).''')
    parser.add_argument('--last', metavar='N', type=int, required=True, help='''
        Id of the last time step (inclusive).''')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Accumulate sparse word co-occurrence matrices over several time steps.''')
    add_cli_args(parser)
    args = parser.parse_args()

    num_timesteps = args.last - args.first + 1
    npos_accum = None
    vocab_size = None
    for timestep in range(args.first, args.last + 1):
        print("Reading time step %d ..." % timestep)
        sys.stdout.flush()
        current = dataset.SingleTimestep(
            args.input % timestep, -1, dense=False, neg_ratio=None)

        if npos_accum is None:
            vocab_size = current.vocab_size
            npos_accum = current.n_pos
        else:
            assert vocab_size == current.vocab_size
            npos_accum += current.n_pos

    del current

    if sys.byteorder == 'big':
        # The file must be written in little endian, so we need to byteswap.
        print("Swapping byte order ...")
        sys.stdout.flush()
        indptr.byteswap(inplace=True)
        indices.byteswap(inplace=True)
        data.byteswap(inplace=True)

    assert npos_accum.nnz < 2**32
    assert npos_accum.indptr.dtype == np.int32
    assert npos_accum.indptr[-1] == npos_accum.nnz
    assert len(npos_accum.indptr) == vocab_size + 1
    assert npos_accum.indices.dtype == np.int32
    assert len(npos_accum.indices) == npos_accum.nnz
    assert npos_accum.data.dtype == np.float32
    assert len(npos_accum.data) == npos_accum.nnz

    print("Writing output to file %s ..." % args.output)
    sys.stdout.flush()
    with gzip.open(args.output, 'wb') as file:
        file.write(struct.pack("<2L", vocab_size, npos_accum.nnz))
        file.write(npos_accum.indptr)
        file.write(npos_accum.indices)
        file.write(npos_accum.data)

    print("Done.")
