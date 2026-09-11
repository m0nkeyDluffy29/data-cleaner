"""
MCP Server for Data & Image Cleaning — VS Code Integration

Run directly:  python mcp_server.py
In VS Code:    started from .vscode/mcp.json (stdio transport)
"""

from __future__ import annotations

import io
import os

import pandas as pd
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from PIL import Image

mcp = FastMCP("DataCleaningServer")


DATA_READ_EXTS = {'.csv', '.json', '.xlsx', '.xls'}
DATA_WRITE_EXTS = {'.csv', '.json', '.xlsx'}   # pandas has no .xls writer
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
JPEG_EXTS = {'.jpg', '.jpeg'}

MAX_PREVIEW_ROWS = 200


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _resolve_input(path: str) -> str:
    """Expand the path and confirm it is a readable file."""
    full = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(full):
        raise ToolError(f"Input file not found: {full}")
    if not os.path.isfile(full):
        raise ToolError(f"Not a file: {full}")
    return full


def _resolve_output(path: str, overwrite: bool) -> str:
    """Expand the path, refuse a silent overwrite, and create the parent dir."""
    full = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(full) and not overwrite:
        raise ToolError(
            f"Output file already exists: {full}. "
            f"Pass overwrite=True to replace it."
        )
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return full


def _drop_index_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Drop a leading unnamed index column, returning (frame, dropped name).

    Files saved with df.to_csv(path) carry the index as a nameless first column,
    which pandas reads back as "Unnamed: 0". Because its values are unique per
    row, leaving it in place makes every row distinct and silently turns
    duplicate detection into a no-op. Only a column that really looks like a
    saved index - unnamed, numeric, complete and unique - is removed.
    """
    if df.columns.empty:
        return df, None
    first = df.columns[0]
    if isinstance(first, str) and first.startswith("Unnamed:"):
        col = df[first]
        if col.notna().all() and col.is_unique and pd.api.types.is_numeric_dtype(col):
            return df.drop(columns=[first]), first
    return df, None


def _read_table(path: str, drop_index: bool = True) -> tuple[pd.DataFrame, str | None]:
    """
    Load a dataframe, dispatching on the file's own extension.

    Returns the frame plus the name of any saved-index column that was removed.
    """
    ext = _ext(path)
    if ext not in DATA_READ_EXTS:
        raise ToolError(
            f"Unsupported input type '{ext or '(no extension)'}'. "
            f"Supported: {', '.join(sorted(DATA_READ_EXTS))}"
        )
    try:
        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext == '.json':
            df = pd.read_json(path)
        else:
            df = pd.read_excel(path)
    except ImportError as exc:
        # pandas defers its Excel engines; .xls in particular needs xlrd.
        extra = " Reading .xls needs xlrd:  pip install xlrd" if ext == '.xls' else ""
        raise ToolError(f"Missing dependency for '{ext}': {exc}.{extra}") from exc
    except Exception as exc:
        raise ToolError(f"Could not read {os.path.basename(path)}: {exc}") from exc

    return _drop_index_column(df) if drop_index else (df, None)


def _write_table(df: pd.DataFrame, path: str) -> None:
    """Save a dataframe in the format named by the OUTPUT path's extension."""
    ext = _ext(path)
    try:
        if ext == '.csv':
            df.to_csv(path, index=False)
        elif ext == '.json':
            df.to_json(path, orient='records', indent=4)
        elif ext == '.xlsx':
            df.to_excel(path, index=False)
        else:  # pragma: no cover - guarded by _check_write_ext
            raise ToolError(f"Unsupported output type '{ext}'")
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not write {os.path.basename(path)}: {exc}") from exc


def _check_write_ext(path: str) -> str:
    """Validate the output extension up front, before any work is done."""
    ext = _ext(path)
    if ext in DATA_WRITE_EXTS:
        return ext
    if ext == '.xls':
        raise ToolError(
            "Cannot write legacy .xls files — pandas dropped that writer. "
            "Use .xlsx instead."
        )
    raise ToolError(
        f"Unsupported output type '{ext or '(no extension)'}'. "
        f"Supported: {', '.join(sorted(DATA_WRITE_EXTS))}"
    )


def _normalize_series(s: pd.Series) -> pd.Series:
    """
    Lower-case and strip a text column so near-identical values compare equal.

    Numeric, boolean and datetime columns are returned untouched. Note that
    pandas 3 gives string columns the dedicated "str" dtype rather than
    "object", so dtype is probed by kind rather than compared directly.
    """
    if (
        pd.api.types.is_numeric_dtype(s)
        or pd.api.types.is_bool_dtype(s)
        or pd.api.types.is_datetime64_any_dtype(s)
    ):
        return s
    try:
        return s.astype("string").str.strip().str.lower()
    except (AttributeError, TypeError, ValueError):
        return s


@mcp.tool()
def clean_data_file(
    input_path: str,
    output_path: str,
    drop_na: str = "any",
    dedupe_on: list[str] | None = None,
    normalize_text: bool = False,
    drop_index_column: bool = True,
    overwrite: bool = False,
) -> str:
    """
    Clean a data file by removing duplicate rows and rows with missing values.

    The output format follows the OUTPUT path's extension, so this doubles as a
    format converter (e.g. .csv in, .json out).

    Args:
        input_path:  Path to the input file (.csv, .json, .xlsx, .xls)
        output_path: Path for the cleaned file (.csv, .json, .xlsx)
        drop_na:     Which rows to drop for missing values:
                     "any"  - drop a row if ANY cell is empty (default)
                     "all"  - drop a row only if EVERY cell is empty
                     "none" - keep rows with missing values
        dedupe_on:   Column names to identify duplicates by, e.g. ["id"].
                     Default (None) compares whole rows.
        normalize_text:
                     Ignore case and surrounding whitespace when comparing, so
                     "Bob", "bob" and " BOB " count as the same value. This
                     affects MATCHING only — the row that is kept keeps its
                     original values (default: False)
        drop_index_column:
                     Discard a leading unnamed index column, the kind left by
                     df.to_csv(path). Keeping it makes every row unique and
                     silently disables duplicate detection (default: True)
        overwrite:   Replace output_path if it already exists (default: False)

    Returns:
        A summary of how many rows were removed and why.

    Example:
        clean_data_file("~/data/raw.csv", "~/data/clean.csv", drop_na="all")
        clean_data_file("~/raw.csv", "~/clean.csv", dedupe_on=["email"],
                        normalize_text=True)
    """
    if drop_na not in ("any", "all", "none"):
        raise ToolError(f"drop_na must be 'any', 'all' or 'none' — got '{drop_na}'")

    src = _resolve_input(input_path)
    _check_write_ext(output_path)
    dst = _resolve_output(output_path, overwrite)

    df, dropped_index = _read_table(src, drop_index=drop_index_column)
    initial_count = len(df)

    if dedupe_on:
        missing = [c for c in dedupe_on if c not in df.columns]
        if missing:
            raise ToolError(
                f"Column(s) not found: {', '.join(missing)}. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )

    key = df[dedupe_on] if dedupe_on else df
    if normalize_text:
        key = key.apply(_normalize_series)
    df = df[~key.duplicated()]
    after_dupes = len(df)

    if drop_na != "none":
        df = df.dropna(how=drop_na)
    final_count = len(df)

    _write_table(df, dst)

    dupes_removed = initial_count - after_dupes
    na_removed = after_dupes - final_count
    matched_on = f"on {dedupe_on}" if dedupe_on else "on whole rows"
    if normalize_text:
        matched_on += ", case/space-insensitive"
    note = (
        f" Dropped saved-index column '{dropped_index}' before comparing rows."
        if dropped_index else ""
    )
    return (
        f"Success! {initial_count} rows in -> {final_count} rows out "
        f"({dupes_removed} duplicates [{matched_on}], "
        f"{na_removed} with missing values [drop_na='{drop_na}']). "
        f"Saved to: {dst}.{note}"
    )


@mcp.tool()
def preview_data_file(input_path: str, rows: int = 5) -> str:
    """
    Inspect a data file before cleaning: shape, null counts, duplicates and a
    sample of the first N rows.

    Args:
        input_path: Path to the file (.csv, .json, .xlsx, .xls)
        rows:       Number of rows to show, 1-200 (default: 5)

    Returns:
        A text summary followed by a table of the first N rows.
    """
    if rows < 1:
        raise ToolError(f"rows must be at least 1 — got {rows}")
    rows = min(rows, MAX_PREVIEW_ROWS)

    src = _resolve_input(input_path)
    df, dropped_index = _read_table(src)

    total_rows = len(df)
    shown = min(rows, total_rows)
    nulls = int(df.isnull().sum().sum())
    dupes = int(df.duplicated().sum())

    note = (
        f"\nIgnoring saved-index column '{dropped_index}' "
        f"(it would mask duplicates)."
        if dropped_index else ""
    )
    return (
        f"File: {os.path.basename(src)}\n"
        f"Rows: {total_rows} | Columns: {len(df.columns)}\n"
        f"Null values: {nulls} | Duplicate rows: {dupes}{note}\n\n"
        f"Preview (first {shown} of {total_rows} rows):\n"
        f"{df.head(rows).to_string(index=False)}"
    )


@mcp.tool()
def list_data_files(folder_path: str) -> str:
    """
    List the supported data files (.csv, .json, .xlsx, .xls) in a folder.

    Args:
        folder_path: Path to the folder to scan (not recursive)

    Returns:
        The matching file names, with sizes.
    """
    full = os.path.abspath(os.path.expanduser(folder_path))
    if not os.path.isdir(full):
        raise ToolError(f"Folder not found: {full}")

    try:
        entries = [
            entry for entry in os.scandir(full)
            if entry.is_file() and _ext(entry.name) in DATA_READ_EXTS
        ]
    except OSError as exc:
        raise ToolError(f"Could not read folder {full}: {exc}") from exc

    if not entries:
        return f"No supported data files found in: {full}"

    listing = "\n".join(
        f"  - {entry.name}  ({entry.stat().st_size / 1024:.1f} KB)"
        for entry in sorted(entries, key=lambda e: e.name)
    )
    return f"Found {len(entries)} file(s) in {full}:\n{listing}"


def _save_image(img: Image.Image, path: str, ext: str) -> bool:
    """
    Save an image in the format named by `ext`.

    JPEG has no alpha channel, so transparency is flattened onto white.
    Returns True if that flattening happened.
    """
    flattened = False
    params: dict[str, object] = {}

    if ext in JPEG_EXTS:
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            canvas = Image.new("RGB", rgba.size, (255, 255, 255))
            canvas.paste(rgba, mask=rgba.split()[-1])
            img = canvas
            flattened = True
        params = {"quality": 85, "optimize": True}
    elif ext == '.webp':
        params = {"quality": 85}
    elif ext == '.png':
        params = {"optimize": True}

    img.save(path, **params)
    return flattened


@mcp.tool()
def clean_image_file(
    input_path: str,
    output_path: str,
    remove_bg: bool = False,
    overwrite: bool = False,
) -> str:
    """
    Optimize an image, and optionally strip its background with AI.

    The output format follows the OUTPUT path's extension, so this also
    converts between formats (e.g. .png in, .webp out).

    Note: remove_bg=True downloads a ~176 MB model on its very first use,
    which can take several minutes. Later calls are fast. Prefer a .png or
    .webp output there — JPEG cannot store transparency and will be
    flattened onto a white background.

    Args:
        input_path:  Path to the source image (.jpg, .jpeg, .png, .webp)
        output_path: Path for the result (.jpg, .jpeg, .png, .webp)
        remove_bg:   Cut out the background using rembg (default: False)
        overwrite:   Replace output_path if it already exists (default: False)

    Returns:
        A message naming the file written and its size.

    Example:
        clean_image_file("~/pics/cat.jpg", "~/pics/cat.png", remove_bg=True)
    """
    src = _resolve_input(input_path)
    in_ext, out_ext = _ext(src), _ext(output_path)

    if in_ext not in IMAGE_EXTS:
        raise ToolError(
            f"Unsupported input image '{in_ext or '(no extension)'}'. "
            f"Supported: {', '.join(sorted(IMAGE_EXTS))}"
        )
    if out_ext not in IMAGE_EXTS:
        raise ToolError(
            f"Unsupported output image '{out_ext or '(no extension)'}'. "
            f"Supported: {', '.join(sorted(IMAGE_EXTS))}"
        )

    dst = _resolve_output(output_path, overwrite)

    try:
        if remove_bg:
            # Imported lazily: rembg pulls in onnxruntime and costs ~2s,
            # which every server start would otherwise pay for.
            from rembg import remove

            with open(src, 'rb') as handle:
                cutout = remove(handle.read())
            with Image.open(io.BytesIO(cutout)) as img:
                flattened = _save_image(img, dst, out_ext)
        else:
            with Image.open(src) as img:
                flattened = _save_image(img, dst, out_ext)
    except ToolError:
        raise
    except ImportError as exc:
        raise ToolError(f"Background removal needs rembg: pip install rembg ({exc})") from exc
    except Exception as exc:
        raise ToolError(f"Image processing failed: {exc}") from exc

    size_kb = os.path.getsize(dst) / 1024
    action = "Background removed" if remove_bg else "Image optimized"
    warning = (
        "  (warning: transparency was flattened onto white — "
        "use .png or .webp to keep it)" if flattened and remove_bg else ""
    )
    return f"Success: {action} and saved as {out_ext} ({size_kb:.1f} KB) at {dst}{warning}"


if __name__ == "__main__":
    mcp.run()
