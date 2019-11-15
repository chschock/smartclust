from setuptools import setup, find_packages

# setup metainfo
libinfo_content = open("__init__.py", "r").readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith("__version__")][0]
exec(version_line)  # produce __version__

setup(
    name="smartclust",
    version=__version__,
    description="Project hierarchical cluster tree onto plausible flat clusters.",
    url="https://github.com/chschock/smartclust",
    long_description=open("Readme.md", "r").read(),
    long_description_content_type="text/markdown",
    author="Christoph Schock",
    author_email="chschock@gmail.com",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requires=["scipy"],
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: MIT License",
        "Operating System :: Linux",
        "Topic :: Scientific/Engineering",
    ),
    keywords="hierarchical clustering",
)
