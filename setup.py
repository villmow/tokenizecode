from distutils.core import setup

setup(name='tokenizecode',
      packages=['tokenizecode'],
      version='0.2',
      python_requires='>=3.9',
      description='Easy code parsing/tokenization to be used in machine learning models',
      author='Johannes Villmow',
      author_email='johannes.villmow@hs-rm.de',
      url='https://github.com/villmow/tokenizecode',
      # download_url='https://github.com/villmow/tokenizecode/archive/refs/tags/v0.1.tar.gz',
      license='MIT License',
      entry_points={
            # 'console_scripts': [
            #       'train_tree_sentencepiece.py',
            # ],
      },
      install_requires=[
            "tree_sitter",
            "requests ",
            "python-magic",
            "tensortree",
            "tokenizers",
      ],
      keywords=['source code', 'tokenization', 'parsing', 'code parser'],  # Keywords that define your package best
      classifiers=[
            'Development Status :: 4 - Beta',  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
            'Intended Audience :: Developers',  # Define that your audience are developers
            'Topic :: Software Development',
            'Topic :: Software Development :: Documentation',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT',
            'Programming Language :: Python :: 3',
      ],
)