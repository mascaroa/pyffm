import pathlib
import setuptools

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setuptools.setup(
    name="pyFFM",
    version="0.0.7",
    author="Aaron Mascaro",
    author_email="mascaroa1@gmail.com",
    description="Python implementation of Factorization Machines (+ Field Aware)",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/mascaroa/pyffm",
    packages=setuptools.find_packages(),
    install_requires=["numpy~=1.18", "pandas~=1.4", "numba~=0.55"],
)
