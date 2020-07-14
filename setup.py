from setuptools import setup, find_namespace_packages

install_requires = ['numpy', 'jax', 'jaxlib']

setup(
    name='nlmax',
    version='0.0.1',
    packages=find_namespace_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    url='',
    license='',
    author='Clement Gehring',
    author_email='clement@gehring.io',
    description='',
    install_requires=install_requires,
)
