from pathlib import Path
from setuptools import setup, find_packages

curr_file = Path(__file__).parent.resolve()


def parse_requirements(filename):
    print(filename)
    return [line for line in Path(filename).read_text().splitlines() if line]


setup(
    name="reflex-llms",
    version="0.2.0",
    author="Amadou Wolfgang Cisse",
    author_email="amadou.6e@googlemail.com",
    description=
    "Python package for autofallback to local LLMs if OpenAI API or Azure Endpoint is not available",
    long_description_content_type="text/markdown",
    url="https://github.com/amadou-6e/docker-db.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires=parse_requirements("requirements.txt"),
    include_package_data=True,
    packages=find_packages(exclude=["usage", "usage.*"]),
    data_files=[(".", ["requirements.txt"])],
    package_data={
        "reflex_llms": ["configs/*/*", "requirements.txt"],
    },
)
