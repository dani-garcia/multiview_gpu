import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raises
import pytest

import multiview_gpu as mv_gpu
from multiview_gpu.util import load_data_np, load_data_tf, load_data_tf_internal

import multiview as mv_cpu

# @pytest.mark.parametrize(bench_names, bench_params)
bench_names = "k, num_mat, mat_length"
# These benchmark parameters are tested on an NVIDIA GTX 1080 with 8GB of VRAM
# Using a card with less memory may require lowering the parameters.
bench_params = [
    (25, 3, 100),
    (25, 3, 500),
    (25, 3, 1000),
    (25, 3, 1500),

    (25, 2, 1000),
    # (25, 3, 1000), # Repeat
    (25, 4, 1000),
    (25, 5, 1000),
    (25, 6, 1000),

    (10, 3, 1500),
    # (25, 3, 1500), # Repeat
    (70, 3, 1500),
    (100, 3, 1500),
    (150, 3, 1500),
    
    # To compare with initial results
    (100, 2, 1500)
]

"""
------ CPC ------
"""


@pytest.mark.parametrize(bench_names, bench_params)
def test_cpc_bench_gpu(sess, benchmark, k, num_mat, mat_length):
    benchmark.group = "cpc-gpu"
    # or benchmark.group = "cpc[{}]".format(k)
    data = load_data_tf_internal(num_mat, mat_length, mat_length)

    def operation(): return sess.run(mv_gpu.mvcpc.cpc(data, k=k))
    benchmark(operation)


@pytest.mark.parametrize(bench_names, bench_params)
def test_cpc_bench_cpu(benchmark, k, num_mat, mat_length):
    benchmark.group = "cpc-cpu"
    data = load_data_np(num_mat, mat_length, mat_length)

    def operation(): return mv_cpu.cpcmv.cpc(data, k=k)
    benchmark(operation)


"""
------ MDS ------
"""


@pytest.mark.parametrize(bench_names, bench_params)
def test_mds_bench_gpu(sess, benchmark, k, num_mat, mat_length):
    benchmark.group = "mds-gpu"
    data = load_data_tf(num_mat, mat_length, mat_length)
    is_distance = [False] * num_mat

    def operation(): return sess.run(mv_gpu.mvmds.mvmds(data, is_distance, k=k))
    benchmark(operation)


@pytest.mark.parametrize(bench_names, bench_params)
def test_mds_bench_cpu(benchmark, k, num_mat, mat_length):
    benchmark.group = "mds-cpu"
    data = load_data_np(num_mat, mat_length, mat_length)
    is_distance = [False] * num_mat

    def operation(): return mv_cpu.mvmds.mvmds(data, is_distance, k=k)
    benchmark(operation)


"""
------ SC ------
"""


@pytest.mark.parametrize(bench_names, bench_params)
def test_sc_bench_gpu(sess, benchmark, k, num_mat, mat_length):
    benchmark.group = "sc-gpu"
    data = load_data_tf(num_mat, mat_length, mat_length)
    is_distance = [False] * num_mat

    def operation(): return sess.run(mv_gpu.mvsc.mvsc(data, is_distance, k=k))
    benchmark(operation)


@pytest.mark.parametrize(bench_names, bench_params)
def test_sc_bench_cpu(benchmark, k, num_mat, mat_length):
    benchmark.group = "sc-cpu"
    data = load_data_np(num_mat, mat_length, mat_length)
    is_distance = [False] * num_mat

    def operation(): return mv_cpu.mvsc.mvsc(data, is_distance, k=k)
    benchmark(operation)
