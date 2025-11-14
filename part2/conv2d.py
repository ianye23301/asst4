import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape

    out_channels_ = bias.shape[0]
    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = (input_height - filter_height + 1)
    out_width = (input_width - filter_width + 1)

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    assert in_channels % 128 == out_channels % 128 == 0
    assert nl.tile_size.gemm_moving_fmax >= out_width


    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )


    max_partition = nl.tile_size.pmax
    c_in_tiles = in_channels // max_partition
    c_out_tiles = out_channels // max_partition

    TILE_SIZE = 128

    bias_buf = nl.zeros((TILE_SIZE, c_out_tiles), dtype=nl.float32, buffer=nl.sbuf)

    for out_t in nl.affine_range(c_out_tiles):
        out_t_start = out_t * TILE_SIZE
        out_t_end = (out_t + 1) * TILE_SIZE
        nisa.dma_copy(src=bias[out_t_start:out_t_end], dst=bias_buf[:,out_t])


    for out_t in nl.affine_range(c_out_tiles):
        out_t_start = out_t * TILE_SIZE
        out_t_end = (out_t + 1) * TILE_SIZE


        W_packed = nl.ndarray(
            shape=(TILE_SIZE, filter_height, filter_width, c_out_tiles, c_in_tiles, TILE_SIZE),
            dtype=W.dtype,
            buffer=nl.sbuf,
        )
        w_big = nl.zeros((TILE_SIZE, in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=W[out_t_start:out_t_end, :, :, :], dst=w_big)


        for in_t in nl.affine_range(c_in_tiles):
            in_t_start = in_t * TILE_SIZE
            in_t_end = (in_t + 1) * TILE_SIZE

            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):

                    w_src = w_big[:,in_t_start:in_t_end,i,j]
                    w_trn = nisa.nc_transpose(w_src, engine=nisa.vector_engine)
                    nisa.dma_copy(src=w_trn, dst=W_packed[:, i, j, out_t, in_t, :])


        #main comp
        for b in nl.affine_range(batch_size):

            for p_row in nl.affine_range(out_pool_height):

                rows_buffer = nl.ndarray((TILE_SIZE, out_width), dtype=nl.float32, buffer=nl.sbuf)

                x_big = nl.ndarray(shape=(TILE_SIZE, input_width), dtype=X.dtype, buffer=nl.sbuf)

                accum = nl.zeros((TILE_SIZE, out_width), dtype=nl.float32,buffer=nl.psum)

                for p in nl.sequential_range(pool_size):
                    row = p_row * pool_size + p
                    accum[:] = nisa.tensor_scalar(accum, nl.multiply, 0.0, dtype=nl.float32)

                    for i in nl.sequential_range(filter_height):
                        for in_t in nl.sequential_range(c_in_tiles):

                            in_t_start = in_t * TILE_SIZE
                            in_t_end = (in_t + 1) * TILE_SIZE

                            nisa.dma_copy(src=X[b, in_t_start:in_t_end, row + i, :], dst=x_big)

                            for j in nl.sequential_range(filter_width):

                                x_buf = x_big[:, j : j + out_width]
                                w_buf = W_packed[:, i, j, out_t, in_t, :]
                                accum += nisa.nc_matmul(w_buf, x_buf)
                    if p == 0:
                        rows_buffer[:] = accum
                    else:
                        rows_buffer[:] =  nl.maximum(rows_buffer, accum)
                bias_tile = bias_buf[:, out_t]
                rows_buffer[:] = nisa.tensor_scalar(rows_buffer, nl.add, bias_tile)

                if pool_size > 1:
                    rows_pooled = rows_buffer.reshape((TILE_SIZE, out_pool_width, pool_size))
                    cols_pooled = nl.max(rows_pooled, axis=2)  # Shape: (TILE_SIZE, out_pool_width)
                    nl.store( X_out[b, out_t_start:out_t_end, p_row, :],value= cols_pooled)
                else:
                    nl.store( X_out[b, out_t_start:out_t_end, p_row, :],value=rows_buffer)

    return X_out
