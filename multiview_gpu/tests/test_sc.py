import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_almost_equal as array_eq
from sklearn.utils.testing import assert_raises
import pytest

import multiview_gpu.mvsc as mvsc
from multiview_gpu.util import load_data_tf, load_data_tf_internal, normalize_labels


def test_laplacian_ng(sess):
    data = np.arange(25, dtype=float).reshape((5, 5))
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    laplacian = sess.run(mvsc.laplacian_ng(data))

    real_laplacian = np.array([[0., 0.0534, 0.0816, 0.1028, 0.1206],
                               [0.2672, 0.1714, 0.1527, 0.1466, 0.1450],
                               [0.4082, 0.2400, 0.2, 0.1820, 0.1723],
                               [0.5144, 0.2933, 0.2380, 0.2117, 0.1964],
                               [0.6030, 0.3384, 0.2708, 0.2378, 0.2181]])

    array_eq(laplacian, real_laplacian, decimal=4)


def test_suggested_sigma(sess):
    data = np.arange(25, dtype=float).reshape((5, 5))
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    s_sigma = sess.run(mvsc.suggested_sigma(data))

    real_s_sigma = 7.0

    array_eq(s_sigma, real_s_sigma, decimal=4)


def test_gaussian_similarity(sess):
    data = np.arange(25, dtype=float).reshape((5, 5))
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    similarity = sess.run(mvsc.distance_gaussian_similarity(data, 2))

    real_similarity = np.array([[1., 0.8824, 0.6065, 0.3246, 0.1353],
                                [0.0439, 0.0110, 0.0021, 0.0003, 0.],
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.]])

    array_eq(similarity, real_similarity, decimal=4)


def test_mvsc_error():
    # Data and is_distane do not have the same length.
    one = np.arange(25, dtype=float).reshape((5, 5))
    two = np.arange(25, 50, dtype=float).reshape((5, 5))
    data = [one, two]
    is_distance = [False, False, False]

    mvsc_est = mvsc.MVSC(k=2)
    assert_raises(ValueError, mvsc_est.fit, data, is_distance)

    # Sample data matrices do not have the same number of rows
    one = np.arange(25, dtype=float).reshape((5, 5))
    two = np.arange(25, 49, dtype=float).reshape((4, 6))
    data = [one, two]
    is_distance = [False, False]

    mvsc_est = mvsc.MVSC(k=2)
    assert_raises(ValueError, mvsc_est.fit, data, is_distance)

    # k value cannot be negative
    one = np.arange(25, dtype=float).reshape((5, 5))
    two = np.arange(25, 50, dtype=float).reshape((5, 5))
    data = [one, two]
    is_distance = [False, False]

    mvsc_est = mvsc.MVSC(k=-2)
    assert_raises(ValueError, mvsc_est.fit, data, is_distance)


# These test results come from the original multiview library
test_names = "data, is_distance, k, real_clust, real_evalues, real_evectors, real_sigmas"
test_params = [
    (
        np.arange(50, dtype=float).reshape((2, 5, 5)),
        [False] * 2,
        2,
        np.array([1, 1, 0, 0, 0]),
        np.array([[1., 1.],  [0.77826244, 0.77826244]]),
        np.array([[5.74599544e-01, -8.18434704e-01],
                  [7.53383370e-01, -6.57581553e-01],
                  [1.00000000e+00,  9.22146033e-17],
                  [7.53383370e-01,  6.57581553e-01],
                  [5.74599544e-01,  8.18434704e-01]]),
        np.array([11.180339887498949, 11.180339887498949]),
    ),
    (
        np.array([[[2, 1, 8], [4, 5, 6], [3, 7, 9]],
                  [[1, 4, 7], [2, 5, 8], [3, 6, 9]]]),
        [False] * 2,
        3,
        np.array([0, 1, 2]),
        np.array([[0.999787863, 0.9997750024],
                  [0.3626654354, 0.4917126113],
                  [0.1521310784, 0.1085691341]]),
        np.array([[-0.5514153943,  0.7561708137, -0.3523446656],
                  [-0.6124669089, -0.0801844847,  0.7864189303],
                  [-0.566414467, -0.6494429528, -0.5073445601]]),
        np.array([4.127431419704746, 1.7320508075688774])
    ),
    (
        np.array([[[2, 1, 8], [4, 5, 6], [3, 7, 9]],
                  [[1, 4, 7], [2, 5, 8], [3, 6, 9]]]),
        [True, False],
        3,
        np.array([1, 0, 2]),
        np.array([[0.9963515948,  0.9979235367],
                  [-0.0415010992,  0.1392406325],
                  [-0.0460637924,  0.4628925786]]),
        np.array([[-0.6018213661,  0.6073008357,  0.5186489547],
                  [-0.602940189, -0.7713714183,  0.2035909219],
                  [-0.5237119168,  0.1901889321, -0.8303938814]]),
        np.array([5.333333333333333, 1.7320508075688774])
    ),
    (
        np.arange(108, dtype=float).reshape((3, 6, 6)),
        [False] * 3,
        2,
        np.array([1, 1, 1, 0, 0, 0]),
        np.array([[1., 1., 1.],
                  [0.843652721, 0.843652721, 0.843652721]]),
        np.array([[0.5720156674,  0.8202426935],
                  [0.6886947786,  0.7250513788],
                  [0.9303987309,  0.366549044],
                  [0.9303987309, -0.366549044],
                  [0.6886947786, -0.7250513788],
                  [0.5720156674, -0.8202426935]]),
        np.array([14.69693845669907, 14.69693845669907, 14.69693845669907])
    ),
    (
        np.arange(144, dtype=float).reshape((4, 6, 6)),
        [True, False, False, True],
        3,
        np.array([2, 1, 1, 1, 0, 0]),
        np.array([[2.4376666738,  0.9764337919,  0.9764337919,  0.893712437],
                  [-1.930572853,  0.8467851846,  0.8467851846,  0.1038947808],
                  [0.3824273906,  0.5165703811,  0.5165703811,  0.0010473741]]),
        np.array([[0.6383932658,  0.5325610732, -0.5557272186],
                  [0.8137737582,  0.356507887,  0.4589928072],
                  [0.6216468995, -0.1718141703,  0.7642218416],
                  [0.5535709189, -0.6639364289,  0.5027401477],
                  [0.4392785844, -0.8820686749, -0.1702620863],
                  [0.2704821226, -0.668974934, -0.6923235942]]),
        np.array([8.5, 14.69693845669907, 14.69693845669907, 116.5])
    ),
    (
        np.arange(256, dtype=float).reshape((4, 8, 8)),
        [False] * 4,
        5,
        np.array([3, 2, 2, 1, 1, 0, 0, 4]),
        np.array([[1., 1., 1., 1.],
                  [0.8616146349, 0.8616146349, 0.8616146349, 0.8616146349],
                  [0.5772379564, 0.5772379564, 0.5772379564, 0.5772379564],
                  [0.3177209061, 0.3177209061, 0.3177209061, 0.3177209061],
                  [0.1449991583, 0.1449991583, 0.1449991583, 0.1449991583]]),
        np.array([[0.3275529033, -0.4783527002,  0.4995628835,  0.4863038858, 0.4217027931],
                  [0.5039855942, -0.6195823133,  0.3350768153, -0.0884398528, -0.4919534512],
                  [0.4869341169, -0.4075217004, -0.1994241714, -0.6065980109, -0.4348448949],
                  [0.5088375772, -0.1516113958, -0.6104607341, -0.3696419355, 0.4569473015],
                  [0.5088375772,  0.1516113958, -0.6104607341,  0.3696419355, 0.4569473015],
                  [0.4869341169,  0.4075217004, -0.1994241714,  0.6065980109, -0.4348448949],
                  [0.5039855942,  0.6195823133,  0.3350768153,  0.0884398528, -0.4919534512],
                  [0.3275529033,  0.4783527002,  0.4995628835, -0.4863038858, 0.4217027931]]),
        np.array([28.284271247461902, 28.284271247461902, 28.284271247461902, 28.284271247461902])
    )

]


@pytest.mark.parametrize(test_names, test_params)
def test_mvsc_multiple(sess, data, is_distance, k, real_clust, real_evalues, real_evectors, real_sigmas):
    data_tf = tf.convert_to_tensor(data, dtype=tf.float64)

    result = sess.run(mvsc.mvsc(data_tf, is_distance, k=k))
    (clust, evalues, evectors, sigmas) = result

    # from multiview.mvsc import mvsc as sc_cpu
    # print("Multiview Result")
    # np.set_printoptions(precision=10, suppress=True)
    # print(sc_cpu(data, is_distance, k=k))

    array_eq(normalize_labels(clust), normalize_labels(real_clust), decimal=4)
    array_eq(evalues, real_evalues, decimal=4)
    array_eq(np.abs(evectors), np.abs(real_evectors), decimal=4)
    array_eq(sigmas, real_sigmas, decimal=4)
