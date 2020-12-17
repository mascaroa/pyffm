import setuptools

setuptools.setup(name="pyFFM",
                 version="0.0.1",
                 author="Aaron Mascaro",
                 author_email="mascaroa1@gmail.com",
                 url="https://github.com/mascaroa/pyffm",
                 packages=setuptools.find_packages(),
                 install_requires=['numpy>=1.16',
                                   'pandas>=0.24',
                                   'numba>=0.49.1']
                 )
