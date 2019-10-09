import io
import os
import re
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

# Package arguments
PACKAGE_NAME = "tabnet"
SHORT_DESCRIPION = "Tensorflow 2.0 implementation of TabNet of any configuration."
URL = "https://github.com/titu1994/tf-TabNet"
LICENCE = 'MIT'

# Extra requirements and configs
EXTRA_REQUIREMENTS = {
    'cpu': ['tensorflow'],
    'gpu': ['tensorflow-gpu']
}
REQUIRED_PYTHON = ">=3.0.0"  # Can be None, or a string value

# Signature arguments
AUTHOR = "Somshubra Majumdar"
EMAIL = "titu1994@gmail.com"


###############################################################

base_path = os.path.abspath(os.path.dirname(__file__))

if LICENCE is None or LICENCE == '':
    raise RuntimeError("Licence must be provided !")

if os.path.exists(os.path.join(base_path, 'LICENCE')):
    raise RuntimeError("Licence must be provided !")


def get_version():
    """Return package version as listed in `__version__` in `init.py`."""
    init_py = open(os.path.join(PACKAGE_NAME, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


try:
    with open(os.path.join(base_path, 'requirements.txt'), encoding='utf-8') as f:
        REQUIREMENTS = f.read().split('\n')

except Exception:
    REQUIREMENTS = []

try:
    with io.open(os.path.join(base_path, 'README.md'), encoding='utf-8') as f:
        LONG_DESCRIPTION = '\n' + f.read()

except FileNotFoundError:
    LONG_DESCRIPTION = SHORT_DESCRIPION


class UploadCommand(Command):
    description = 'Build, install and upload tag to git with cleanup.'
    user_options = []

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(base_path, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel'.format(sys.executable))

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(get_version()))
        os.system('git push --tags')

        try:
            self.status('Removing build artifacts...')
            rmtree(os.path.join(base_path, 'build'))
            rmtree(os.path.join(base_path, 'tabnet.egg-info'))
        except OSError:
            pass

        sys.exit()

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    @staticmethod
    def status(s):
        print(s)


setup(
    name=PACKAGE_NAME,
    version=get_version(),
    packages=find_packages(exclude=['tests']),
    url=URL,
    download_url=URL,
    python_requires=REQUIRED_PYTHON,
    license=LICENCE,
    author=AUTHOR,
    author_email=EMAIL,
    description=SHORT_DESCRIPION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ),
    test_suite="tests",
    # python setup.py upload
    cmdclass={
        'upload': UploadCommand,
    },
)
