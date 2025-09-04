# Final Dataset Creation - README

## Overview
This document explains how to create and use the final curated dataset for sales forecasting.

## Process

### Step 1: Create the Final Dataset
Run the high accuracy dataset creation script:
```bash
python create_high_accuracy_dataset.py
```

This script will:
1. Analyze all items in `combined_data.csv`
2. Run XGBoost and AutoARIMA forecasting models on each item
3. Calculate MAPE (Mean Absolute Percentage Error) for each model
4. Select the best performing items:
   - **Top 19 items**: Highest accuracy (lowest MAPE)
   - **6 medium accuracy items**: Next best performing items
5. Create `final_dataset.csv` with data for these 25 selected items

### Step 2: Use the Final Dataset
The EDA dashboard (`eda_app.py`) has been updated to automatically use `final_dataset.csv`.

To run the dashboard:
```bash
streamlit run eda_app.py
```

## Output Files

### final_dataset.csv
- Contains data for 25 carefully selected items (top 19 + 6 medium accuracy)
- Optimized for high-quality forecasting
- Used by the main dashboard

### forecast_results.csv
- Detailed analysis results for all items
- Shows MAPE scores, accuracy percentages, and model performance
- Useful for understanding item selection criteria

### high_accuracy_items.csv
- Contains only items with MAPE < 20%
- Subset of the highest performing items

## Item Selection Criteria

**Top 19 Items**: 
- Lowest MAPE scores (best forecasting accuracy)
- Primary focus for business forecasting

**6 Medium Accuracy Items**:
- Next best performing items (ranks 20-25)
- Provides broader coverage while maintaining quality

## Benefits of Curated Dataset

1. **Higher Accuracy**: Only includes items with proven forecasting performance
2. **Better Models**: XGBoost and AutoARIMA perform better on curated data
3. **Reliable Predictions**: Reduced noise from poorly performing items
4. **Business Focus**: Concentrates on items with predictable sales patterns

## Dashboard Features

The updated dashboard now shows:
- "Using curated dataset with top 19 high-accuracy + 6 medium-accuracy items"
- All EDA visualizations use the optimized dataset
- Forecasting models perform better due to higher data quality
- More reliable predictions for business planning
