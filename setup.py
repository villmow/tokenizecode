from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop


class PostInstall(install):
    """This downloads the tree-sitter grammars at the end of the installation process."""

    def run(self):
        install.run(self)
        print("Downloading tree-sitter grammars... (this may take a while)", flush=True)
        from tokenizecode import CodeParser

        t = CodeParser()  # just to download the grammars
        print("Done!")


class PostDevelop(develop):
    """This downloads the tree-sitter grammars at the end of the installation process."""

    def run(self):
        develop.run(self)

        print("Downloading tree-sitter grammars... (this may take a while)", flush=True)
        from tokenizecode import CodeParser

        t = CodeParser()  # just to download the grammars
        print("Done!")


setup(
    name="tokenizecode",
    packages=["tokenizecode"],
    version="0.2",
    python_requires=">=3.9",
    description="Easy code parsing/tokenization to be used in machine learning models",
    author="Johannes Villmow",
    author_email="johannes.villmow@hs-rm.de",
    url="https://github.com/villmow/tokenizecode",
    # download_url='https://github.com/villmow/tokenizecode/archive/refs/tags/v0.1.tar.gz',
    license="MIT License",
    entry_points={
        # 'console_scripts': [
        #       'train_tree_sentencepiece.py',
        # ],
    },
    install_requires=[
        "tree-sitter==0.20.4",
        "requests ",
        "python-magic",
        "tensortree",
        "tokenizers",
        "transformers",
        "datasets",
    ],
    keywords=[
        "source code",
        "tokenization",
        "parsing",
        "code parser",
    ],  # Keywords that define your package best
    classifiers=[
        "Development Status :: 4 - Beta",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development",
        "Topic :: Software Development :: Documentation",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT",
        "Programming Language :: Python :: 3",
    ],
    cmdclass={
        "install": PostInstall,
        "develop": PostDevelop,
    },
)
