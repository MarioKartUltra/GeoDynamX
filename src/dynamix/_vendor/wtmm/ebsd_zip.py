"""On-the-fly access to EBSD datasets stored as zip archives.

Lets you keep large EBSD datasets (.ctf, .cpr/.crc) compressed on disk and
read them transparently — no permanent unzipped copy.

Two modes:

1. **Enumerate**: ``list_ebsd_files(EBSD_DIR)`` walks both expanded folders
   AND zip archives in the directory, returning a flat list of "virtual"
   paths.  Real files use a normal Path; entries inside a zip use the
   :class:`ZipEntry` wrapper which prints as ``archive.zip!entry/path.ctf``.

2. **Read**: ``open_ebsd(virtual_path)`` is a context manager that yields:
     - a real file path (str) for use with binary loaders that need seeking
       (e.g. ``read_cpr``).  For zip entries this extracts to a tempfile that
       is auto-deleted on exit.
     - For ASCII formats like .ctf you can also use ``read_text(path)`` to
       get the entire file as a string with no temp-file detour.

Design choice: keep the existing ``read_ctf(path)`` and ``read_cpr(path)``
APIs unchanged — they accept ordinary path strings.  This module only adds
two utilities (file listing + open-as-real-path) so the calling notebook
needs minimal changes.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

# Extensions we treat as EBSD files (case-insensitive)
EBSD_EXTENSIONS = ('.ctf', '.cpr')


# ---------------------------------------------------------------------------
# Virtual path wrapper
# ---------------------------------------------------------------------------

class ZipEntry:
    """Represents a single file inside a zip archive.  Acts like Path enough
    that downstream code that calls ``.suffix``, ``.name``, ``.stem`` works.

    Format: ``ZipEntry(zip_path, entry_name)``.
    String repr: ``"<zip_path>!<entry_name>"`` (so it shows up nicely in lists).
    """
    __slots__ = ('zip_path', 'entry')

    def __init__(self, zip_path, entry):
        self.zip_path = Path(zip_path)
        self.entry = str(entry)

    def __str__(self):
        return f'{self.zip_path}!{self.entry}'
    __repr__ = __str__
    def __fspath__(self):                    # so os.fspath / Path can wrap us
        return str(self)

    @property
    def suffix(self):
        return Path(self.entry).suffix
    @property
    def name(self):
        return Path(self.entry).name
    @property
    def stem(self):
        return Path(self.entry).stem
    @property
    def parts(self):
        # Used by the notebook's relative_to logic.  Treat the zip itself as
        # the "containing folder" so relative tags still work.
        return (self.zip_path.name,) + Path(self.entry).parts

    def relative_to(self, other):
        """Mimic Path.relative_to.  ``self`` lives inside the zip which is
        inside ``other`` (typically EBSD_DIR), so we return a fake Path
        spelled as <zip_stem>/<entry>."""
        try:
            zip_rel = self.zip_path.relative_to(other)
        except ValueError:
            zip_rel = Path(self.zip_path.name)
        return zip_rel.with_suffix('') / self.entry

    def is_inside_zip(self):
        return True


def is_zip_entry(p):
    return isinstance(p, ZipEntry)


# ---------------------------------------------------------------------------
# Enumeration: walk a directory, expand zip contents
# ---------------------------------------------------------------------------

def list_zip_ebsd_entries(zip_path, extensions=EBSD_EXTENSIONS):
    """List all .ctf/.cpr entries inside a single zip archive.  Returns a
    list of ZipEntry objects (or [] if the zip is unreadable / empty)."""
    zip_path = Path(zip_path)
    out = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                ext = Path(info.filename).suffix.lower()
                if ext in extensions:
                    out.append(ZipEntry(zip_path, info.filename))
    except (zipfile.BadZipFile, OSError) as e:
        print(f'  WARN: cannot read {zip_path}: {e}')
    return out


def list_ebsd_files(ebsd_dir, extensions=EBSD_EXTENSIONS, sort=True):
    """Walk ``ebsd_dir`` and return a flat list of EBSD files.  Both real
    files (via rglob) AND files inside .zip archives are included.

    Each entry is either a Path (for an unzipped file) or a ZipEntry (for an
    inside-zip file).  Both expose .suffix, .name, .relative_to so existing
    selector code keeps working.

    If a dataset exists in BOTH forms (an unzipped folder side-by-side with
    a same-named zip), the unzipped version takes priority and the zipped
    duplicates are skipped — so you can have a partially-extracted setup
    without seeing each file twice.
    """
    ebsd_dir = Path(ebsd_dir)
    real_files = []
    for ext in extensions:
        real_files.extend(ebsd_dir.rglob(f'*{ext}'))
    real_files = sorted(real_files)
    real_set = {p.resolve() for p in real_files}

    zip_entries = []
    for zp in sorted(ebsd_dir.glob('*.zip')):
        # If a sibling folder of the same stem exists with extracted files,
        # skip this zip's CTFs/CPRs to avoid duplicates.
        sibling = zp.with_suffix('')
        if sibling.is_dir() and any(sibling.rglob(f'*{ebs}') for ebs in extensions):
            continue
        zip_entries.extend(list_zip_ebsd_entries(zp, extensions=extensions))

    out = real_files + zip_entries
    if sort:
        out.sort(key=lambda p: (str(p).lower()))
    return out


# ---------------------------------------------------------------------------
# Reading: get a real file path or in-memory text
# ---------------------------------------------------------------------------

@contextmanager
def open_ebsd(path, binary_for_cpr=True):
    """Context manager that yields a real filesystem path for ``path``.

    - If ``path`` is a regular file, yields its path unchanged.
    - If ``path`` is a :class:`ZipEntry`, extracts the entry to a temp dir.
      For ``.cpr`` entries the matching ``.crc`` companion in the same zip
      is also extracted (CPR readers seek into the .crc binary alongside
      the .cpr text header).  Temp files are deleted on exit.

    Usage
    -----
    >>> with open_ebsd(virtual_path) as real_path:
    ...     ebsd = read_ctf(real_path)   # or read_cpr(real_path)
    """
    if not isinstance(path, ZipEntry):
        yield str(path)
        return

    # Extract to a temp directory so paired files (e.g. .cpr + .crc) keep
    # their shared stem.  We use the original entry's basename so that
    # read_cpr's ``filepath.with_suffix('.crc')`` lookup finds the companion.
    tmp_dir = tempfile.mkdtemp(prefix='ebsd_zip_')
    main_basename = Path(path.entry).name
    main_stem = Path(path.entry).stem
    main_suffix = Path(path.entry).suffix.lower()

    def _extract_one(zf, entry_name, dst_path):
        with zf.open(entry_name) as src, open(dst_path, 'wb') as dst:
            while True:
                buf = src.read(1 << 20)   # 1 MiB chunks
                if not buf: break
                dst.write(buf)

    extracted = []
    try:
        with zipfile.ZipFile(path.zip_path, 'r') as zf:
            # Always extract the requested entry
            main_dst = os.path.join(tmp_dir, main_basename)
            _extract_one(zf, path.entry, main_dst)
            extracted.append(main_dst)

            # For .cpr/.crc pairs, also extract the companion if it exists.
            if main_suffix in ('.cpr', '.crc'):
                companion_suffix = '.crc' if main_suffix == '.cpr' else '.cpr'
                # Look for sibling entry in the same archive with same stem
                # in the same path-prefix (handles entries inside subfolders
                # of the zip, e.g. "12644973/Foo.cpr" pairs with
                # "12644973/Foo.crc").
                main_dir = str(Path(path.entry).parent).replace('\\', '/')
                candidates = [
                    f'{main_dir}/{main_stem}{companion_suffix}',
                    f'{main_dir}/{main_stem}{companion_suffix.upper()}',
                    f'{main_stem}{companion_suffix}',
                    f'{main_stem}{companion_suffix.upper()}',
                ]
                # Strip leading "./" and normalize
                names_in_zip = set(zf.namelist())
                companion_entry = None
                for cand in candidates:
                    cand = cand.lstrip('./')
                    if cand in names_in_zip:
                        companion_entry = cand; break
                # Last-resort: case-insensitive scan
                if companion_entry is None:
                    for n in names_in_zip:
                        if (Path(n).stem.lower() == main_stem.lower()
                                and Path(n).suffix.lower() == companion_suffix):
                            companion_entry = n; break
                if companion_entry is not None:
                    comp_basename = f'{main_stem}{companion_suffix}'
                    comp_dst = os.path.join(tmp_dir, comp_basename)
                    _extract_one(zf, companion_entry, comp_dst)
                    extracted.append(comp_dst)
                else:
                    # No companion -- read_cpr will raise FileNotFoundError
                    # below, but we leave that to the reader so the user gets
                    # the standard error message.  Print a diagnostic.
                    print(f'  open_ebsd: WARN: no {companion_suffix} companion '
                          f'for {path.entry!r} in {path.zip_path.name}')
        yield extracted[0]
    finally:
        for p in extracted:
            try: os.unlink(p)
            except OSError: pass
        try: os.rmdir(tmp_dir)
        except OSError: pass


def read_ctf_text(path):
    """Return the entire CTF file as a single string, without writing a
    temp file (works for both real paths and ZipEntries).  Useful when the
    downstream parser is line-by-line and accepts either a path or already-
    read text.
    """
    if isinstance(path, ZipEntry):
        with zipfile.ZipFile(path.zip_path, 'r') as zf:
            with zf.open(path.entry) as f:
                return io.TextIOWrapper(f, encoding='utf-8',
                                         errors='replace').read()
    with open(path, 'r', errors='replace') as f:
        return f.read()


__all__ = [
    'EBSD_EXTENSIONS',
    'ZipEntry',
    'is_zip_entry',
    'list_zip_ebsd_entries',
    'list_ebsd_files',
    'open_ebsd',
    'read_ctf_text',
]
