from setuptools import setup, find_packages

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    name="cms_study",
    version="0.1.0",
    author="cyan",
    author_email="allmon.autumn@example.com",
    # packages=["health_modeling"],
    # scripts=['bin/script1','bin/script2'],
    url="http://pypi.python.org/pypi/PackageName/",
    license="LICENSE.txt",
    description="An awesome package that does something",
    long_description=open("README.md").read(),
    install_requires=[],
)
