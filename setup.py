import os
import setuptools

README = os.path.join(os.getcwd(), 'README.md')

setuptools.setup(name="pyFFM",
                 version="0.0.1",
                 author="Aaron Mascaro",
                 author_email="mascaroa1@gmail.com",
                 description="Python implementation of Factorization Machines (+ Field Aware)",
                 long_description=README,
                 long_description_content_type="text/markdown",
                 license="MIT",
                 url="https://github.com/mascaroa/pyffm",
                 packages=setuptools.find_packages(),
                 install_requires=['numpy>=1.16',
                                   'pandas>=0.24',
                                   'numba>=0.49.1']
                 )
