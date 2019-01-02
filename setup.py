import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multiview_gpu",
    version="0.1.0",
    author="Daniel García García",
    author_email="danigarcia@uoc.edu",
    description="GPU-accelerated multiview clustering and dimensionality reduction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dani-garcia/multiview_gpu",
    keywords=["multiview", "clustering", "dimensionality reduction"],
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
    ],
    extras_require={
        "tf": ["tensorflow>=1.12.0"],
        "tf_gpu": ["tensorflow-gpu>=1.12.0"],
    },
    setup_requires=[
        "pytest-runner"
    ],
    tests_require=[
        "pytest", 
        "pytest-benchmark"
    ],
    classifiers=[
        "Programming Language :: Python",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)