from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install

# For post-install script running using CustomInstallCommand, see:
# https://stackoverflow.com/a/45021666/3592884

import subprocess

def install_jupyter_extensions():
    print('About to run install_jupyter_extensions')

    process = subprocess.Popen(["./install_jupyter_extensions.sh"], stdout=subprocess.PIPE)
    for line in process.stdout:
        print(line.decode('utf8'))
    
    print('install_jupyter_extensions completed')


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        install_jupyter_extensions()

setup(
    name='jupyter_openscad',
    version='0.1.0',
    description='Jupyter renderer for the OpenSCAD & SolidPython constructive solid geometry systems',
    author='Nick Choly',
    author_email="nickcholy@gmail.com",
    url='https://github.com/nickc92/JupyterOpenSCAD',
    py_modules=['jupyter_openscad'],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
    packages=find_packages(),
    install_requires=['jupyterlab', 'ipywidgets', 'pythreejs', 'solidpython'],
)
