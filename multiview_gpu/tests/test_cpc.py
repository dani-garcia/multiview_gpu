import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_almost_equal as array_eq
from sklearn.utils.testing import assert_raises
import pytest

import multiview_gpu.mvcpc as mvcpc
from multiview_gpu.util import load_data_np, load_data_tf_internal

# These test results come from the original multiview library
test_names = "data, k, real_evalues, real_evectors"
test_params = [
    (
        np.arange(50, dtype=float).reshape((2, 5, 5)),
        2,
        np.array([[66.60954262, 176.89090037],
                  [-6.60954262, 8.10909963]]),

        np.array([[0.20303518, 0.74751369],
                  [0.31154745, 0.45048661],
                  [0.42005972, 0.15345953],
                  [0.528572, -0.14356755],
                  [0.63708427, -0.44059463]])
    ),
    (
        np.array([[[2, 1, 8], [4, 5, 6], [3, 7, 9]],
                  [[1, 4, 7], [2, 5, 8], [3, 6, 9]]]),
        3,
        np.array([[16.09601677,  16.21849616],
                  [-0.11903382,  -0.85516505],
                  [0.02301705,  -0.3633311]]),
        np.array([[0.45139369, -0.88875921,  0.07969196],
                  [0.55811719,  0.35088538,  0.75192065],
                  [0.69623914,  0.29493478, -0.65441923]]),
    ),
    (
        np.arange(108, dtype=float).reshape((3, 6, 6)),
        1,
        np.array([[115.9976632412, 316.1710085402, 516.3443538393]]),
        np.array([[-0.231240774],
                  [-0.2959473864],
                  [-0.3606539987],
                  [-0.4253606111],
                  [-0.4900672235],
                  [-0.5547738359]])
    ),
    (
        np.arange(256, dtype=float).reshape((4, 8, 8)),
        2,
        np.array([[275.4408167097,  761.3336991414, 1247.226581573,
                   1733.1194640047],
                  [-23.4408167097,    2.6663008586,   28.773418427,
                   54.8805359953]]),
        np.array([[0.2224698261, -0.6059487133],
                  [0.2573131708, -0.455630833],
                  [0.2921565154, -0.3053129526],
                  [0.3269998601, -0.1549950723],
                  [0.3618432048, -0.004677192],
                  [0.3966865494,  0.1456406883],
                  [0.4315298941,  0.2959585686],
                  [0.4663732387,  0.4462764489]])
    )
]


@pytest.mark.parametrize(test_names, test_params)
def test_cpc_multiple(sess, data, k, real_evalues, real_evectors):
    data_tf = tf.convert_to_tensor(data, dtype=tf.float64)

    evalues, evectors = sess.run(mvcpc.cpc(data_tf, k=k))

    # from multiview.cpcmv import cpc as cpc_cpu
    # print("Multiview Result")
    # np.set_printoptions(precision=10, suppress=True)
    # print(cpc_cpu(data, k=k))

    array_eq(evalues, real_evalues, decimal=4)
    array_eq(np.abs(evectors), np.abs(real_evectors), decimal=4)


def test_mvcpc_error():
    data = np.arange(40, dtype=float).reshape((2, 5, 4))

    # k value cannot be negative
    cpc_est = mvcpc.MVCPC(k=-2)
    assert_raises(ValueError, cpc_est.fit, data)

    # Second and third dimensions must be equal
    cpc_est = mvcpc.MVCPC(k=2)
    assert_raises(ValueError, cpc_est.fit, data)


def test_mvcpc_fit():
    data = load_data_np(2, 5, 5)

    cpc_est = mvcpc.MVCPC(k=2)
    cpc_est.fit_transform(data)
