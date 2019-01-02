"""
CPC computes Common principal components of a set of matrices.

This file uses a variation of Trendafilov (2010) method to compute
the k first common principal components of a set of matrices in an
efficient way
"""

import warnings

from sklearn.base import BaseEstimator
import scipy.sparse.linalg as sparse

import tensorflow as tf

from multiview_gpu.util import cond_less


def cpc(x, k=0):
    dtype = x.dtype
    
    n_g = tf.fill([x.shape[0]], tf.cast(x.shape[1], dtype))
    p = x.shape[1]
    mcas = x.shape[0]# 
    n = n_g / tf.reduce_sum(n_g)

    D = tf.TensorArray(dtype=dtype, size=k)
    CPC = tf.TensorArray(dtype=dtype, size=k)
    Qw = tf.eye(p.value, dtype=dtype)

    s0 = tf.zeros([p, p], dtype=dtype)
    s = tf.foldl(lambda acc, nx: acc + nx[0] * nx[1],
                 (n, x), initializer=s0)

    res_vectors = eig_vectors(s, k, p)
    q0 = tf.reverse(res_vectors, [1])

    output = tf.while_loop(cond_less(k),
                           _for_ncomp_in_k,
                           [0, D, CPC, k, q0, mcas, x, p, n_g, Qw])

    D_final = output[1].stack()
    CPC_final = tf.transpose(output[2].stack())

    return D_final, CPC_final


def eig_vectors(s, k, p):
    return tf.cond(tf.equal(k, p),
                   # TODO This gives different signs in CPU than GPU
                   lambda: tf.self_adjoint_eig(s)[1],
                   # NOTE: This path is not GPU optimized
                   # TODO: This outputs garbage values when run twice in tests?
                   # TODO: Try removing the tf.cond and use a normal if
                   lambda: sparse_eigsh_tf(s, k))


def sparse_eigsh_tf(s, k):
    return tf.py_func(lambda x: sparse.eigsh(x, k=k)[1], [s], s.dtype, stateful=False)


def _for_ncomp_in_k(ncomp, D, CPC, k, q0, mcas, x, p, n_g, Qw):
    q = q0[:, ncomp]
    q = tf.reshape(q, [-1, 1])
    d = calculate_d(x, q)

    # Second for-loop
    iterator = 15
    output = tf.while_loop(cond_less(iterator),
                           _for_in_iterator,
                           [0, q, d, p, mcas, ncomp, n_g, x, Qw])
    q, d = output[1:3]

    # Final part
    D = D.write(ncomp, d)
    CPC = CPC.write(ncomp, q[:, 0])
    Qw -= tf.matmul(q, q, transpose_b=True)

    return tf.add(ncomp, 1), D, CPC, k, q0, mcas, x, p, n_g, Qw


def calculate_d(x, q):
    def fn(x_m): return tf.matmul(tf.matmul(q, x_m, transpose_a=True), q)
    return tf.squeeze(tf.map_fn(fn, x))


def _for_in_iterator(i, q, d, p, mcas, ncomp, n_g, x, Qw):
    p = x.shape[1]
    s0 = tf.zeros([p, p], dtype=x.dtype)
    s = tf.foldl(lambda acc, vars: acc + (vars[0] * vars[1] / vars[2]),
                 (n_g, x, d), initializer=s0)

    w = tf.matmul(s, q)
    w = tf.cond(tf.not_equal(ncomp, 0),
                lambda: tf.matmul(Qw, w),
                lambda: w)

    q = w / tf.sqrt(tf.matmul(w, w, transpose_a=True))

    d = calculate_d(x, q)

    return tf.add(i, 1), q, d, p, mcas, ncomp, n_g, x, Qw


class MVCPC(BaseEstimator):
    """Compute common principal components of x.

    Parameters
    ----------

    k : int, default 0
        Number of components to extract (0 means all p components).

    Attributes
    ----------
    eigenvalues_ : ndarray
        Stores the eigenvalues computed in the algorithm.
    eigenvectors_ : ndarray
        Stores the eigenvectors computed in the algorithm.

    References
    ----------

        Trendafilov, N. (2010). Stepwise estimation of common principal
        components. *Computational Statistics and Data Analysis*, 54,
        3446–3457.
    """

    def __init__(self, k=0):
        self.k = k

    def fit(self, x):
        """Compute k common principal components of x.

        Parameters
        ----------

        x : array_like or ndarray
            A set of n matrices of dimension pxp given as a n x p x p  matrix.

        """
        self.fit_transform(x)
        return self

    def fit_transform(self, x):
        """Compute k common principal components of x, and return those
        components.

        Parameters
        ----------

        x : array_like or ndarray
            A set of n matrices of dimension pxp given as a n x p x p  matrix.

        Returns
        -------
        values : tuple
            Tuple with two elements:

            the eigenvalues

            the common eigenvectors

        Raises
        ------

            ValueError: Matrices are not square matrices or k value is
            negative.

        Examples
        --------

        >>> import tensorflow as tf
        >>> x = tf.convert_to_tensor(([[[2, 1, 8], [4, 5, 6], [3, 7, 9]],
                      [[1, 4, 7], [2, 5, 8], [3, 6, 9]]])
        >>> mv_cpc = MVCPC(k=3)
        >>> mv_cpc.fit_transform(x)
        (array([[ 16.09601677,  16.21849616],
                [ -0.11903382,  -0.85516505],
                [  0.02301705,  -0.3633311 ]]),
                array([[ 0.45139369, -0.88875921,  0.07969196],
                [ 0.55811719,  0.35088538,  0.75192065],
                [ 0.69623914,  0.29493478, -0.65441923]]))
        >>>
        """

        if x.shape[1] != x.shape[2]:
            raise ValueError("matrices have different size from m x n x n. "
                             "Size found instead is {} {} {}".format(*x.shape))
        if self.k == 0:
            # If k is 0 then retrieve all the components
            self.k = x.shape[1]
        elif self.k > x.shape[1]:
            self.k = x.shape[1]
            warnings.warn("k is greater than matrix dimension. Maximum "
                          "possible number of components is computed instead.")
        elif self.k < 0:
            raise ValueError("k value must be between 0 and number of samples"
                             " of data matrix.")

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.Session() as sess:
            D, CPC = sess.run(cpc(x, self.k))

        self.eigenvalues_ = D
        self.eigenvectors_ = CPC
        return (self.eigenvalues_, self.eigenvectors_)
