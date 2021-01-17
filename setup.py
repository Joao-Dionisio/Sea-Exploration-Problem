import setuptools

NAME = "seaexp"

VERSION = "0.2101.0"

AUTHOR = 'João Pedro Pedroso, João Pedro Dionísio and Davi Pereira-Santos'

AUTHOR_EMAIL = 'dpsabc@gmail.com'

DESCRIPTION = 'Sea Exploration Problem'

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()

LICENSE = 'GPL3'

URL = 'https://github.com/Joao-Dionisio/Sea-Exploration-Problem'

DOWNLOAD_URL = 'https://github.com/Joao-Dionisio/Sea-Exploration-Problem/releases'

CLASSIFIERS = ['Intended Audience :: Science/Research',
               'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
               'Natural Language :: English',
               'Programming Language :: Python',
               'Topic :: Scientific/Engineering',
               # posix               'Operating System :: Linux',
               'Programming Language :: Python :: 3.8']

INSTALL_REQUIRES = [
    'numpy', 'sklearn', 'lange', 'hyperopt', 'scipy'
]

EXTRAS_REQUIRE = {
}

SETUP_REQUIRES = ['wheel']

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    download_url=DOWNLOAD_URL,
    extras_require=EXTRAS_REQUIRE,
    install_requires=INSTALL_REQUIRES,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    packages=setuptools.find_packages(),
    setup_requires=SETUP_REQUIRES,
    url=URL,
    # scripts=['seaexp']
)

package_dir = {'': 'seaexp'}  # So IDE (e.g. Intellij) can recognize the working dir when running with ctr+shift+F10.
