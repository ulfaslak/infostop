import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="infostop",
    version="0.0.13",
    author="Ulf Aslak",
    author_email="ulfjensen@gmail.com",
    description="Detect stop locations in time-ordered (lat, lon) location data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ulfaslak/infostop",
    packages=setuptools.find_packages(),
    install_requires=[
        'infomap',
        'numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
)
