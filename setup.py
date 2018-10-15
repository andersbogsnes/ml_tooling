from setuptools import setup, find_packages
import os

NAME = 'ml_utils'
DESCRIPTION = 'A library for machine learning utilities'
URL = 'https://lspgitlab01.alm.brand.dk/abanbn/ab_models'
EMAIL = 'abanbn@almbrand.dk'
AUTHOR = 'Anders Bogsnes'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

REQUIRED = [
    'scikit-learn',
    'scipy',
    'pandas',
    'numpy',
    'matplotlib',
    'gitpython'
]

TESTS_REQUIRED = [
    'pytest',
    'pytest-cov'
]

here = os.path.abspath(os.path.dirname(__file__))
about = {}

if not VERSION:
    with open(os.path.join(here, 'src', NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name=NAME,
    version=about['__version__'],
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=REQUIRED,
    tests_require=TESTS_REQUIRED,
    license='MIT',
    package_data={'': ['*.mplstyle']},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='ml framework tooling'
)
