import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_almost_equal as array_eq
from sklearn.utils.testing import assert_raises
import pytest

import multiview_gpu.mvmds as mvmds
from multiview_gpu.util import load_data_tf


def test_preprocess_mds(sess):
    data = np.arange(25, dtype=float).reshape((5, 5))
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    preprocessed_data = sess.run(mvmds.preprocess_mvmds(data))

    sim = np.array([[40., 20., 0., -20., -40.],
                    [20., 10., 0., -10., -20.],
                    [0., 0., 0., 0., 0.],
                    [-20., -10., 0., 10., 20.],
                    [-40., -20., 0., 20., 40.]])

    array_eq(preprocessed_data, sim, decimal=4)


def test_mvmds_error():
    # Data and is_distane do not have the same length.
    one = np.arange(25, dtype=float).reshape((5, 5))
    two = np.arange(25, 50, dtype=float).reshape((5, 5))
    data = [one, two]
    is_distance = [False, False, False]

    mvmds_est = mvmds.MVMDS(k=2)
    assert_raises(ValueError, mvmds_est.fit, data, is_distance)

    # Sample data matrices do not have the same number of rows
    one = np.arange(25, dtype=float).reshape((5, 5))
    two = np.arange(25, 49, dtype=float).reshape((4, 6))
    data = [one, two]
    is_distance = [False, False]

    mvmds_est = mvmds.MVMDS(k=2)
    assert_raises(ValueError, mvmds_est.fit, data, is_distance)

    # k value cannot be negative
    one = np.arange(25, dtype=float).reshape((5, 5))
    two = np.arange(25, 50, dtype=float).reshape((5, 5))
    data = [one, two]
    is_distance = [False, False]

    mvmds_est = mvmds.MVMDS(k=-2)
    assert_raises(ValueError, mvmds_est.fit, data, is_distance)


# These test results come from the original multiview library
test_names = "data, is_distance, k, real_result"
test_params = [
    (
        np.arange(50, dtype=float).reshape((2, 5, 5)),
        [False] * 2,
        2,
        np.array([[-0.632455532,  -0.1989703693],
                  [-0.316227766,  -0.6963962924],
                  [-0.,     -0.3305190213],
                  [0.316227766,   0.0994851846],
                  [0.632455532,  -0.5969111078]])
    ),
    (
        np.array([[[2, 1, 8], [4, 5, 6], [3, 7, 9]],
                  [[1, 4, 7], [2, 5, 8], [3, 6, 9]]]),
        [False] * 2,
        3,
        np.array([[-0.740466335, 0.344058532, 0.5773502692],
                  [0.0722697384, -0.8132919227,  0.5773502692],
                  [0.6681965966, 0.4692333907, 0.5773502692]])
    ),
    (
        np.array([[[2, 1, 8], [4, 5, 6], [3, 7, 9]],
                  [[1, 4, 7], [2, 5, 8], [3, 6, 9]]]),
        [False, False],
        3,
        np.array([[-0.740466335, 0.344058532, 0.5773502692],
                  [0.0722697384, -0.8132919227, 0.5773502692],
                  [0.6681965966,  0.4692333907, 0.5773502692]])
    ),
    (
        np.arange(108, dtype=float).reshape((3, 6, 6)),
        [False] * 3,
        2,
        np.array([[0.5976143047,  0.6346897855],
                  [0.3585685828,  0.1020481616],
                  [0.1195228609,  0.0049779591],
                  [-0.1195228609, -0.0049779591],
                  [-0.3585685828, -0.1020481616],
                  [-0.5976143047, 0.759138763]])
    ),
    (
        np.arange(256, dtype=float).reshape((4, 8, 8)),
        [False] * 4,
        5,
        np.array([[0.5400617249, 0.4806639344, -0.3528596134, 0.3516817362, -0.3523045507],
                  [0.3857583749, 0.1413371427, -0.1352081249, 0.1420639924, -0.1416263226],
                  [0.2314550249, -0.045950973, -0.0591191469, 0.0594984884, -0.0601554582],
                  [0.077151675, 0.0746392244, -0.0825078346, 0.0821366762, -0.0818364084],
                  [-0.077151675, -0.3293040024, 0.3377397826, -0.3387822177,  0.3380904857],
                  [-0.2314550249, 0.1219703099, -0.1324274795, 0.1317680694, -0.1318591591],
                  [-0.3857583749, 0.7860987651, -0.8372055327, 0.8368890363, -0.8369952644],
                  [-0.5400617249, 0.0058598224, 0.1199497301, -0.1154634166,  0.1151282954]])
    )

]


@pytest.mark.parametrize(test_names, test_params)
def test_mvmds_multiple(sess, data, is_distance, k, real_result):
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)

    result = sess.run(mvmds.mvmds(data_tf, is_distance, k=2))

    from multiview.mvmds import mvmds as mds_cpu
    print("Multiview Result")
    np.set_printoptions(precision=10, suppress=True)
    print(mds_cpu(data, is_distance, k=k))

    array_eq(np.abs(result[:, 0]), np.abs(real_result[:, 0]), decimal=4)
