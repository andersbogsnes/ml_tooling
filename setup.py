from setuptools import setup, find_packages
import os

NAME = 'ab_models'
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


setup(
    name=NAME,
    version=about['__version__'],
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=REQUIRED,
    tests_require=TESTS_REQUIRED,
    license='MIT',
    package_data={'': ['*.mplstyle']}
)
