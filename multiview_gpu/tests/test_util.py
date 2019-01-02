import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raises
import multiview_gpu.util as util


def test_hbeta(sess):
    data = np.arange(25, dtype=float).reshape((5, 5))
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    H, P = sess.run(util.Hbeta(data, 2))

    real_H = 100.14586747777895
    real_P = np.array([[-0.15651764, -1.02118236, -1.138202, -1.15403889, -1.15618218],
                       [-1.15647224, -1.1565115, -
                           1.15651681, -1.15651753, -1.15651763],
                       [-1.15651764, -1.15651764, -
                           1.15651764, -1.15651764, -1.15651764],
                       [-1.15651764, -1.15651764, -
                           1.15651764, -1.15651764, -1.15651764],
                       [-1.15651764, -1.15651764, -1.15651764, -1.15651764, -1.15651764]])

    assert_array_almost_equal(H, real_H, decimal=4)
    assert_array_almost_equal(P, real_P, decimal=4)


def _test_whiten(sess):
    data = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3], [1, 3, 2, 4]])
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    whitened = sess.run(util.whiten(data, n_comp=4))

    real_whitened = np.array([[9.63475981e-01, 1.11961253e+00, 1.49011612e-08,
                               0.00000000e+00],
                              [-1.55893688e+00, 6.91958598e-01, 0.00000000e+00,
                               0.00000000e+00],
                              [-1.84007539e-01, -1.46559183e+00,
                               -1.49011612e-08, 0.00000000e+00],
                              [7.79468442e-01, -3.45979299e-01, 0.00000000e+00,
                               0.00000000e+00]])

    assert_array_almost_equal(whitened, real_whitened, decimal=0)
