# CSV Data Combination Tool

## Overview
This script combines data from multiple CSV files, removes duplicates, and creates a single consolidated CSV file for sales forecasting analysis.

## Files Processed
- `filtered_data4.csv`
- `filtered_data5.csv` 
- `filtered_data55.csv`

## Usage
```bash
python combine_csv_data.py
```

## Output
- **File**: `combined_data.csv`
- **Content**: Deduplicated sales data from all source files
- **Columns**: 
  - Posting Date
  - Item No.
  - Description
  - Quantity
  - Unit Price
  - DISCOUNT
  - NET TOTAL

## Process Summary
1. **Read**: Loads all three CSV files
2. **Combine**: Merges all data into a single dataset
3. **Deduplicate**: Removes duplicate records based on all relevant columns
4. **Save**: Outputs clean data to `combined_data.csv`

## Results from Last Run
- **Total records processed**: 273,085
- **Duplicates removed**: 183,519
- **Final unique records**: 89,566
- **Date range**: 01.01.22 to 31.12.24
- **Unique items**: 56
- **Total quantity**: 598,934 units
- **Total net value**: $11,471,028.86

## Integration with Dashboard
The output file `combined_data.csv` can be used directly with the sales forecasting dashboard (`eda_app.py`) for comprehensive analysis and forecasting.
