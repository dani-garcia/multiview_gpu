# Multiview-GPU

GPU-accelerated multiview clustering and dimensionality reduction thanks to Tensorflow.

Based on [multiview](https://pypi.org/project/multiview/).

# Dependedencies
For building:
- tensorflow-gpu
- numpy
- sklearn
- scipy
- setuptools

For testing and benchmarking:
- pytest
- pytest-benchmark
- multiview

```sh
pip install tensorflow-gpu numpy sklearn scipy setuptools pytest pytest-benchmark multiview
```

To run the tests:
```sh
pytest
# or
python setup.py test
```

Note: the benchmarks take some time to run, to skip them:
```sh
pytest --benchmark-disable
```

Conversely, to only run the tests
```sh
pytest --benchmark-only
```
