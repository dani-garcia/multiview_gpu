import tensorflow as tf
import pytest

@pytest.fixture(scope="function")
def sess():
    with tf.Session() as sess:
        yield sess  # provide the fixture value