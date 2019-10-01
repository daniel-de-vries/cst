from setuptools import setup

exec(open('cst/version.py').read())

setup(
    name='cst',
    version=__version__,
    description='Class-Shape Transformation (CST)',
    author='Daniel de Vries',
    author_email="danieldevries6@gmail.com",
    packages=['cst'],
    install_requires=['numpy', 'scipy'],
    url='https://github.com/daniel-de-vries/cst',
    download_url='https://github.com/daniel-de-vries/cst/v{0}.tar.gz'.format(__version__),
    keywords=['optimization', 'class', 'shape', 'transformation', 'mathematics'],
    license='MIT License',
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: Microsoft :: Windows',
                 'Programming Language :: Python :: 3.7'],
)