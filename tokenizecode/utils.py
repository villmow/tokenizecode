from typing import Union, Optional

from pathlib import Path
import re

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


whitespace_replacemap = {
    " ": "·",
    "\t": "↹",
    "\v": "↦",
    "\n": "⏎",
    "\r": "↵",
}
whitespace_restoremap = {v: k for k, v in whitespace_replacemap.items()}


RE_REPLACE_WHITESPACE = re.compile(
    '|'.join(
        sorted(
            re.escape(k) for k in whitespace_replacemap
        )
    )
)
RE_RESTORE_WHITESPACE = re.compile(
    '|'.join(
        sorted(
            re.escape(k) for k in whitespace_restoremap
        )
    )
)


def replace_whitespace(text: Union[str, list[str]], replace_map: dict[str, str] = None):
    """
    Replaces tabs, newlines and spaces with unicode symbols
    """

    def _replace(string: str) -> str:
        return RE_REPLACE_WHITESPACE.sub(
            lambda m: whitespace_replacemap[m.group()],
            string
        )

    if isinstance(text, (list, tuple)):
        res = [_replace(string) for string in text]
        return text.__class__(res)

    return _replace(text)


def restore_whitespace(text: str) -> str:
    return  RE_RESTORE_WHITESPACE.sub(
            lambda m: whitespace_restoremap[m.group()],
            text
        )


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
