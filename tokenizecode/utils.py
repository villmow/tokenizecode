from typing import Union, Optional

from pathlib import Path

from tensortree import TensorTree


HAS_MAGIC = False
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    import warnings
    warnings.warn("Please install python-magic to determine file encoding (requires libmagic"
                  "to be available on your system).")


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def smart_read_file(filepath: Union[str, Path]) -> str:
    """
    Tries to determine the encoding of the file and open it accordingly.
    :param filepath:
    :return:
    """
    def _read_file(filepath: Path, encoding: Optional[str]) -> str:
        with filepath.open("rt", encoding=encoding) as f:
            return f.read()

    filepath = Path(filepath)

    if not HAS_MAGIC:
        return _read_file(filepath)

    # determine encoding
    m = magic.Magic(mime_encoding=True)
    encoding = m.from_file(str(filepath))

    try:
        content = _read_file(filepath, encoding)
    except (UnicodeDecodeError, LookupError):
        # we get an encoding that open() doesn't know: LookupError: unknown encoding: unknown-8bit
        # --> try another time with utf-8 (default encoding)
        # print(f"[LookupError] Could not parse file: {filepath} -> parsing with 'utf-8'")
        content = _read_file(filepath, encoding=None)

    return content


def is_tree_of_strings(tree: TensorTree) -> bool:
    return isinstance(tree.node_data[0], str)


TensorTreeWithStrings = TensorTree  # with only strings as nodes
TensorTreeWithInts = TensorTree  # with ints (ids) as nodes
