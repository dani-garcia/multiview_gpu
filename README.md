# Multiview-GPU

This is a simple example package. You can use
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.

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
