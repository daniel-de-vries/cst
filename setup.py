import re

from setuptools import setup

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('cst/__init__.py').read(),
)[0]

setup(
    name="cst",
    version=__version__,
    description="Class-Shape Transformation (CST)",
    author="Daniel de Vries",
    author_email="danieldevries6@gmail.com",
    packages=["cst"],
    package_data={"cst": ["test/*.py", "test/clark-y.dat", "test/propeller.dat"]},
    install_requires=["numpy", "scipy"],
    extras_require={"test": ["parameterized"]},
    url="https://github.com/daniel-de-vries/cst",
    download_url="https://github.com/daniel-de-vries/cst/v{0}.tar.gz".format(
        __version__
    ),
    keywords=["optimization", "class", "shape", "transformation", "mathematics"],
    license="MIT License",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.7",
    ],
)
