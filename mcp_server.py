"""
MCP Server for Data Cleaning — VS Code Integration
Run with:  python mcp_server.py
"""

from fastmcp import FastMCP
from PIL import Image
from rembg import remove
import pandas as pd
import os


mcp = FastMCP("DataCleaningServer")


@mcp.tool()
def clean_data_file(input_path: str, output_path: str) -> str:
    """
    Cleans a data file by removing duplicates and empty rows.

    Args:
        input_path:  Full path to the input file (.csv, .json, .xlsx, .xls)
        output_path: Full path where the cleaned file will be saved

    Returns:
        A summary message with row counts, or an error description.

    Example:
        clean_data_file("/data/raw.csv", "/data/cleaned.csv")
    """
    ext = os.path.splitext(input_path)[1].lower()

    try:
        if ext == '.csv':
            df = pd.read_csv(input_path)
        elif ext == '.json':
            df = pd.read_json(input_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
        else:
            return f"Unsupported file type: '{ext}'. Supported: .csv, .json, .xlsx, .xls"

        initial_count = len(df)
        df = df.drop_duplicates()
        df = df.dropna()
        final_count = len(df)
        removed = initial_count - final_count

        if ext == '.csv':
            df.to_csv(output_path, index=False)
        elif ext == '.json':
            df.to_json(output_path, orient='records', indent=4)
        elif ext in ['.xlsx', '.xls']:
            df.to_excel(output_path, index=False)

        return (
            f"Success! {initial_count} rows in → {final_count} rows out "
            f"({removed} removed). Saved to: {output_path}"
        )

    except Exception as e:
        return f" Processing failed: {str(e)}"


@mcp.tool()
def preview_data_file(input_path: str, rows: int = 5) -> str:
    """
    Preview the first N rows of a data file before cleaning.

    Args:
        input_path: Full path to the file (.csv, .json, .xlsx, .xls)
        rows:       Number of rows to preview (default: 5)

    Returns:
        A string table of the first N rows.
    """
    ext = os.path.splitext(input_path)[1].lower()

    try:
        if ext == '.csv':
            df = pd.read_csv(input_path)
        elif ext == '.json':
            df = pd.read_json(input_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
        else:
            return f"Unsupported file type: '{ext}'"

        total_rows = len(df)
        nulls = df.isnull().sum().sum()
        dupes = df.duplicated().sum()

        summary = (
            f"File: {os.path.basename(input_path)}\n"
            f"Rows: {total_rows} | Columns: {len(df.columns)}\n"
            f"Null values: {nulls} | Duplicate rows: {dupes}\n\n"
            f"Preview (first {min(rows, total_rows)} rows):\n"
            f"{df.head(rows).to_string(index=False)}"
        )
        return summary

    except Exception as e:
        return f"Preview failed: {str(e)}"


@mcp.tool()
def list_data_files(folder_path: str) -> str:
    """
    List all supported data files (.csv, .json, .xlsx, .xls) in a folder.

    Args:
        folder_path: Path to the folder to scan

    Returns:
        A list of supported data files found in the folder.
    """
    supported = {'.csv', '.json', '.xlsx', '.xls'}

    if not os.path.isdir(folder_path):
        return f" Folder not found: {folder_path}"

    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not files:
        return f"No supported data files found in: {folder_path}"

    file_list = "\n".join(f"  • {f}" for f in sorted(files))
    return f" Found {len(files)} file(s) in {folder_path}:\n{file_list}"

@mcp.tool()
def clean_image_file(input_path: str, output_path: str, remove_bg: bool = False):

    valid_exts = ['.jpg', '.jpeg', '.png', '.webp']
    ext = os.path.splitext(input_path)[1].lower()
    out_ext = os.path.splitext(output_path)[1].lower()

    if ext not in valid_exts:
        return f"Error: Unsupported file {ext}"
    try:
        with open(input_path, 'rb') as f:
            input_data = f.read()
            img = Image.open(input_path)

        if remove_bg:
            output_data = remove(input_data)
            with open(output_path, 'wb') as o:
                o.write(output_data)
            return f"Sucess: Background removed and saved to {output_path}"
        if out_ext in ['.jpg', 'jpeg'] and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.save(output_path, optimize=True, quality=85)
        return f"Success: Image optimized and saved as {out_ext} at {output_path}"

    except Exception as e:
        return f"Image processing failed: {str(e)}"


if __name__ == "__main__":
    mcp.run()