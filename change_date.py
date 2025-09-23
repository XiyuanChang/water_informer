import pandas as pd
from datetime import datetime

def convert_date_format(date_str):
    # Convert '1/5/1982' to '1982-01-05'
    return datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')

def update_csv_date_format(file_paths):
    for file_path in file_paths:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Check if 'date' column exists
        if 'Date' in df.columns:
            # Apply the date format conversion
            df['Date'] = df['Date'].apply(convert_date_format)
            
            # Save the updated DataFrame back to the CSV file
            df.to_csv(file_path, index=False)
            print(f"Updated dates in file: {file_path}")
        else:
            print(f"'date' column not found in file: {file_path}")

# List of file paths to update
file_paths = [
    "climate_new/01331095.csv",
    "climate_new/06441500.csv",
    "climate_new/10265150.csv",
    "climate_washed/01331095.csv",
    "climate_washed/06441500.csv",
    "climate_washed/10265150.csv",
    # Add more file paths as needed
]

# Call the function to update the date format
update_csv_date_format(file_paths)