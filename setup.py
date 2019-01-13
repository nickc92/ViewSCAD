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
#        install_jupyter_extensions()

with open('README.md') as f:
        long_description = f.read()

setup(
    name='viewscad',
    version='0.1.6',
    description='Jupyter renderer for the OpenSCAD & SolidPython constructive solid geometry systems',
    author='Nick Choly',
    author_email="nickcholy@gmail.com",
    url='https://github.com/nickc92/ViewSCAD',
    long_description=long_description,
    long_description_content_type='text/markdown',
    py_modules=['viewscad'],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
    packages=find_packages(),
    install_requires=['jupyter', 'jupyterlab', 'ipywidgets', 'pythreejs', 'solidpython'],
    setup_requires=['jupyter', 'jupyterlab', 'ipywidgets', 'pythreejs', 'solidpython'],
)
