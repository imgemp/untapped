import os
from setuptools import find_packages
from setuptools import setup

version = '0.1.dev1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = open(os.path.join(here, 'CHANGES.rst')).read()
except IOError:
    README = CHANGES = ''

install_requires = [
    'lasagne',
    'Theano',
    'scikit-learn',
    'scipy',
    'numpy',
    'matplotlib'
    ]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    'pytest-pep8',
    ]

setup(
    name="untapped",
    version=version,
    description="Semi^2-Supervised Deep Generative Model",
    long_description="\n\n".join([README, CHANGES]),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Ian Gemp",
    author_email="imgemp@cs.umass.edu",
    url="https://github.com/imgemp/untapped",
    license="MIT",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    setup_requires=['pytest-runner'],
    install_requires=install_requires,
    tests_require=tests_require
    )
