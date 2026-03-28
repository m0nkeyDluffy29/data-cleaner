import pandas as pd
import os
from fastmcp import FastMCP

mcp = FastMCP("DataCleaningServer")

@mcp.tool()
def clean_data_file(input_path: str, output_path: str) -> str:
  """
  Cleans a data file (csv, JSON, xlsx)
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
      return f"Unexpected error in exporting file {ext}"
    
    initial_count = len(df)
    df = df.drop_duplicates()
    df = df.dropna()

    if ext == '.csv':
      df.to_csv(output_path, index=False)
    elif ext == '.json':
      df.to_json(output_path, orient='records', indent=4)
    elif ext == ['.xlsx', '.xls']:
      df.to_excel(output_path, index=False)

    return f"Success! Processed {initial_count} rows. Cleaned file saved to {output_path}."

  except Exception as e:
    return f"Processing Failed: {str(e)}"
  
if __name__ == "__main__":
  mcp.run()
  
  

