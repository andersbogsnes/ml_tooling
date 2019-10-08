import codecs
import os
import re

from setuptools import find_packages, setup

###############################################################################

NAME = "ml_tooling"
PACKAGES = find_packages(where="src")
INSTALL_REQUIRES = [
    "pandas",
    "scikit-learn",
    "matplotlib",
    "pyyaml",
    "gitpython",
    "joblib",
    "sqlalchemy",
    "pyarrow",
    "attrs",
]
KEYWORDS = ["ml", "framework", "tooling"]
PROJECT_URLS = {
    "Documentation": "https://ml-tooling.readthedocs.io/en/stable/",
    "Bug Tracker": "https://github.com/andersbogsnes/ml_tooling/issues",
    "Source Code": "https://github.com/andersbogsnes/ml_tooling",
}
CLASSIFIERS = [
    "Programming Language:: Python:: 3.6",
    "Programming Language:: Python:: 3.7",
    "Programming Language :: Python :: Implementation :: CPython",
    "Intended Audience :: Developers",
    "Development Status:: 3 - Alpha",
    "License:: OSI Approved:: MIT License",
    "Operating System:: OS Independent",
]

META_PATH = os.path.join("src", NAME, "__init__.py")

EXTRAS_REQUIRE = {"docs": ["sphinx"], "tests": ["coverage", "pytest"]}
EXTRAS_REQUIRE["dev"] = (
    EXTRAS_REQUIRE["tests"] + EXTRAS_REQUIRE["docs"] + ["pre-commit", "tox"]
)

###############################################################################

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


VERSION = find_meta("version")
URL = find_meta("url")
LONG = read("README.md")

setup(
    name=NAME,
    version=VERSION,
    description=find_meta("description"),
    url=URL,
    long_description=LONG,
    long_description_content_type="text/markdown",
    author=find_meta("author"),
    author_email=find_meta("email"),
    license=find_meta("license"),
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    zip_safe=False,
    python_requires=">=3.6",
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    package_dir={"": "src"},
    package_data={"": ["*.mplstyle"]},
    extras_require=EXTRAS_REQUIRE,
    project_urls=PROJECT_URLS,
    maintainer=find_meta("author"),
    maintainer_email=find_meta("email"),
)
