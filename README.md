# MultiDataCleaner

MultiDataCleaner is a Python-based data and image cleaning toolkit, designed to streamline the process of cleaning, optimizing, and preparing datasets and images for machine learning and data analysis workflows. It provides both data file cleaning and advanced image processing tools, including AI-powered background removal.

## Features

- **Data Cleaning**: Remove duplicates and empty rows from CSV, JSON, and Excel files.
- **Image Cleaning**: Optimize images (JPG, PNG, WEBP) and optionally remove backgrounds using AI.
- **Extensible Tools**: Easily add new cleaning tools using the FastMCP framework.
- **Command-line Interface**: Run cleaning operations directly from the terminal.

## Requirements

- Python 3.8+
- pip (Python package manager)
- Recommended: virtual environment (venv)

### Python Packages

- pandas
- Pillow
- rembg
- fastmcp

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Data Cleaning

Use the provided tools to clean data files:

- Remove duplicates
- Remove empty rows

### Image Cleaning

Clean and optimize images, with optional background removal:

- Supported formats: JPG, JPEG, PNG, WEBP
- Optionally remove backgrounds using AI

### Example (CLI)

```bash
python mcp_server.py
```

You can then use the registered tools via the FastMCP interface or integrate them into your own workflows.

## File Structure

- `mcp_server.py` — Main server and tool definitions
- `DS.py` — Additional data science utilities (if any)
- `server.py` — (Optional) Alternative server entry point

## Adding New Tools

Add new `@mcp.tool()` functions in `mcp_server.py` to extend the toolkit.

## License

MIT License

## Author

Daksh Sharma
