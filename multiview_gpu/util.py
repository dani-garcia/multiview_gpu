import tensorflow as tf
import numpy as np


def cond_less(val):
    return lambda i, *_: tf.less(i, val)


def tf_print(x):
    return tf.Print(x, [x], summarize=1000)


def length(x):
    t = type(x).__module__

    if t == "tensorflow.python.framework.ops":
        return x.shape[0].value
    if t == "numpy":
        return x.shape[0]
    # Python list
    return len(x)


def to_tensor(x, dtype=tf.float32):
    return tf.convert_to_tensor(x, dtype=dtype)


def to_tensor_list(x, dtype=tf.float32):
    return [to_tensor(elem, dtype=dtype) for elem in x]


def load_data_np(num_mat, x_size, y_size):
    total = num_mat * x_size * y_size
    return np.arange(total, dtype=float).reshape((num_mat, x_size, y_size))


def load_data_tf(num_mat, x_size, y_size, dtype=tf.float32):
    data_np = load_data_np(num_mat, x_size, y_size)
    return to_tensor_list(data_np, dtype=dtype)


def load_data_tf_internal(num_mat, x_size, y_size, dtype=tf.float32):
    data_np = load_data_np(num_mat, x_size, y_size)
    return to_tensor(data_np, dtype=dtype)


def _index_of(value, arr):
    for i, e in enumerate(arr):
        if e == value:
            return i
    return -1


def normalize_labels(arr):
    """
    Normalize the labels returned by clustering algorithms like Kmeans, to be able to compare them.

    r1 = [0, 0, 1, 1, 1]
    r2 = [1, 1, 0, 0, 0]

    assert_array_almost_equal(normalize_labels(r1), normalize_labels(r2), decimal=4)
    """
    labels = []
    result = []

    for label in arr:
        idx = _index_of(label, labels)

        if idx == -1:
            idx = len(labels)
            labels.append(label)
        result.append(idx)
    return result


def euclidean_distances(x):
    """Euclidean distances of a tensor

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor with the euclidean distances of elements of `x`.
    """
    norm = tf.reduce_sum(tf.square(x), 1)

    norm_row = tf.reshape(norm, [-1, 1])
    norm_col = tf.reshape(norm, [1, -1])

    return tf.sqrt(tf.maximum(norm_row + norm_col - 2*tf.matmul(x, x, transpose_b=True), 0.0))


def reduce_var(x, axis, keepdims=False, ddof=0):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.
        ddof : int, optional
            Means Delta Degrees of Freedom. The divisor used in calculations 
            is N - ddof, where N represents the number of elements.
            By default ddof is zero

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    devs_sum = tf.reduce_sum(devs_squared, axis=axis, keep_dims=keepdims)

    # TODO: Might want to change this to allow mutliple axes
    div = tf.cast(tf.shape(x)[axis] - ddof, tf.float32)

    return devs_sum / div


def reduce_std(x, axis, keepdims=False, ddof=0):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.
        ddof : int, optional
            Means Delta Degrees of Freedom. The divisor used in calculations 
            is N - ddof, where N represents the number of elements.
            By default ddof is zero

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims, ddof=ddof))


def Hbeta(D, beta):
    """Compute H beta for matrix D, given beta.

    Parameters
    ----------

    D : ndarray or nscalar.
        Input value.
    beta : numeric.
        Number to operate with in the exponential distribution.

    Returns
    -------

    (H, P) : tuple
        Return a tuple with H and P values. H is a float value and P
        can be either an scalar or a ndarray.

    Examples
    --------

    >>> import numpy as np
    >>> matrix = np.array([[1, 2, 3], [2, 3, 4], [5, 6, 2]])
    >>> hbeta = Hbeta(matrix, 3)
    >>> print("H -> %g\nP-> %a" % hbeta)
        H -> 21.6814
        P-> array([[  8.66214422e-01,   4.31262766e-02,   2.14713088e-03],
                   [  4.31262766e-02,   2.14713088e-03,   1.06899352e-04],
                   [  5.32220535e-06,   2.64977002e-07,   4.31262766e-02]])
    """
    P = tf.exp(-D * beta)
    sumP = tf.reduce_sum(P)

    return tf.cond(tf.equal(sumP, 0),
                   lambda: (0.0, D * 0.0),
                   lambda: (_calculate_h(sumP, beta, D, P), P - sumP)
                   )


def _calculate_h(sumP, beta, D, P):
    return tf.log(sumP) + beta * tf.reduce_sum(tf.matmul(D, P)) / sumP


def whiten(X, n_comp):
    """Whitening of matrix X, and return that new matrix whitened.

    Parameters
    ----------

    X : ndarray
        Input data (2D).
    n_comp : int
        Number of rows of output matrix.

    Returns
    -------

    X : ndarray
        Whitened matrix.

    Examples
    --------

    >>> x = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9])).T
    >>> whiten(x)
        [[  1.22474487e+00  -2.98023224e-08   0.00000000e+00]
         [  0.00000000e+00   0.00000000e+00   0.00000000e+00]
         [ -1.22474487e+00   2.98023224e-08   0.00000000e+00]]
    """
    n = X.shape[0]
    p = X.shape[1]
    # Centering matrix columns
    mean = tf.reduce_mean(X, axis=0)
    # TODO: row_norm parameter ignored
    #sd = reduce_std(X, axis=0, ddof=1)
    X -= mean
    X = tf.transpose(X)

    V = tf.matmul(X, X, transpose_b=True) / tf.cast(n, tf.float32)

    # TODO: This gives slightly different results for u and v
    V = tf.cast(V, tf.float64)
    s, u, v = tf.linalg.svd(V, full_matrices=True, compute_uv=True)
    D = (tf.linalg.diag(tf.rsqrt(s)))
    K = tf.matmul(D, u, transpose_b=True)
    K = tf.cast(K, tf.float32)
    # TODO: Ignoring reshape
    K2 = K[:n_comp, :]
    X = tf.matmul(K2, X, transpose_b=True)
    return X
