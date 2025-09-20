import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

file_name = "Business.xlsx"
sheet_to_load = "Table 1"

# --- 1. Load and clean the data ---
df = pd.read_excel(file_name, sheet_name=sheet_to_load, header=None, skiprows=5)

# Manually fix the column headers
new_headers = df.iloc[0] 
df = df[1:] 
df.columns = new_headers

# Clean and convert the data
columns_to_convert = df.columns[2:]
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')
df.dropna(how='all', inplace=True)

print("--- Data Structure Analysis ---")
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Years available: {list(columns_to_convert)}")
print("\n--- Industry breakdown ---")
print(df.iloc[:, 1].values)  # Print all industry categories

# --- 2. Extract Private sector and calculate Government by subtraction ---
industry_series = df.iloc[:, 1].astype(str)

# Find the private sector row and total space economy row
private_row_idx = None
total_space_economy_row_idx = None

print("\n--- Searching for Private sector and Total Space Economy rows ---")
for idx, industry in enumerate(industry_series):
    print(f"Row {idx}: '{industry.strip()}'")
    
    # Look for Private sector row
    industry_clean = industry.strip().lower()
    if 'private' in industry_clean and ('industries' in industry_clean or 'sector' in industry_clean or industry_clean == 'private'):
        private_row_idx = idx
        print(f"  -> Found Private sector at row {idx}")
    
    # Look for total space economy row (usually the first data row or contains "total" or "space economy")
    if idx == 0 or 'total' in industry_clean or 'space economy' in industry_clean or industry_clean.strip() == '' or 'all industries' in industry_clean:
        if idx == 0:  # First row is usually the total
            total_space_economy_row_idx = idx
            print(f"  -> Found Total Space Economy at row {idx}")

# --- 3. Extract Private values and calculate Government by subtraction ---
if private_row_idx is not None:
    # Extract private sector values
    private_totals = df.iloc[private_row_idx, 2:].copy()
    private_totals = pd.to_numeric(private_totals, errors='coerce')
    private_totals.index = columns_to_convert
    
    print(f"\n--- Extraction Results ---")
    print(f"Private sector row index: {private_row_idx}")
    print(f"Private sector row name: '{df.iloc[private_row_idx, 1]}'")
    print("Private Sector Values:")
    print(private_totals)
    
    # Find total space economy values
    if total_space_economy_row_idx is not None:
        total_space_economy = df.iloc[total_space_economy_row_idx, 2:].copy()
        total_space_economy = pd.to_numeric(total_space_economy, errors='coerce')
        total_space_economy.index = columns_to_convert
        
        print(f"\nTotal space economy row index: {total_space_economy_row_idx}")
        print(f"Total space economy row name: '{df.iloc[total_space_economy_row_idx, 1]}'")
        print("Total Space Economy Values:")
        print(total_space_economy)
    else:
        # If we can't find a clear total row, let's try to sum all rows
        print("\nCould not find clear total row. Calculating total from all data rows...")
        total_space_economy = df.iloc[:, 2:].sum()
        total_space_economy = pd.to_numeric(total_space_economy, errors='coerce')
        total_space_economy.index = columns_to_convert
        print("Calculated Total Space Economy Values:")
        print(total_space_economy)
    
    # Calculate government sector as Total - Private
    government_totals = total_space_economy - private_totals
    
    print("\nGovernment Sector Values (Total - Private):")
    print(government_totals)
    
else:
    print("Could not find Private sector row. Here's what we found:")
    print("Available industry categories:")
    for idx, industry in enumerate(industry_series):
        print(f"  Row {idx}: '{industry.strip()}'")
    
    # Fallback - look for any row containing "private"
    print("\n--- Trying flexible search for Private ---")
    for idx, industry in enumerate(industry_series):
        industry_clean = industry.strip().lower()
        if 'private' in industry_clean:
            print(f"Found potential Private row {idx}: '{industry.strip()}'")
            private_row_idx = idx
            break
    
    if private_row_idx is not None:
        private_totals = pd.to_numeric(df.iloc[private_row_idx, 2:], errors='coerce')
        private_totals.index = columns_to_convert
        
        # Calculate total and government
        total_space_economy = df.iloc[:, 2:].sum()
        total_space_economy = pd.to_numeric(total_space_economy, errors='coerce')
        total_space_economy.index = columns_to_convert
        
        government_totals = total_space_economy - private_totals
        
        print(f"\nUsing Private row {private_row_idx}: '{df.iloc[private_row_idx, 1]}'")
        print("Private Values:", private_totals.values)
        print("Government Values (calculated):", government_totals.values)

# --- 4. Create forecasting function ---
def create_forecast(historical_data, years_ahead=5, model_type='polynomial'):
    """
    Create forecasts using different models
    
    Parameters:
    - historical_data: pandas Series with years as index and values
    - years_ahead: number of years to forecast
    - model_type: 'linear', 'polynomial', or 'exponential'
    """
    years = np.array([int(year) for year in historical_data.index]).reshape(-1, 1)
    values = historical_data.values
    
    # Create future years
    last_year = int(historical_data.index[-1])
    future_years = np.array(range(last_year + 1, last_year + years_ahead + 1)).reshape(-1, 1)
    all_future_years = np.array(range(last_year + 1, last_year + years_ahead + 1))
    
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(years, values)
        predictions = model.predict(future_years)
    
    elif model_type == 'polynomial':
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(years, values)
        predictions = model.predict(future_years)
    
    elif model_type == 'exponential':
        # Fit exponential growth: y = a * e^(b*x)
        log_values = np.log(values)
        model = LinearRegression()
        model.fit(years, log_values)
        log_predictions = model.predict(future_years)
        predictions = np.exp(log_predictions)
    
    return all_future_years, predictions, model

# --- 5. Enhanced visualization with projections ---
def plot_with_forecasts(private_totals, government_totals, forecast_years=5):
    """
    Create comprehensive plots with historical data and forecasts
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Space Economy Analysis: Historical Data and Projections', fontsize=16, y=0.98)
    
    # Historical years for plotting
    years = [int(year) for year in private_totals.index]
    
    # Plot 1: Historical trends
    ax1 = axes[0, 0]
    ax1.plot(years, private_totals.values, marker='o', label='Private Sector', linewidth=2, markersize=6)
    ax1.plot(years, government_totals.values, marker='s', label='Government', linewidth=2, markersize=6)
    ax1.set_title('Historical Trends (Real Value Added)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Millions of Dollars')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Growth rates
    ax2 = axes[0, 1]
    private_growth = private_totals.pct_change() * 100
    government_growth = government_totals.pct_change() * 100
    ax2.plot(years[1:], private_growth.iloc[1:], marker='o', label='Private Sector', linewidth=2)
    ax2.plot(years[1:], government_growth.iloc[1:], marker='s', label='Government', linewidth=2)
    ax2.set_title('Year-over-Year Growth Rates')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Growth Rate (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Forecasts - Linear
    ax3 = axes[1, 0]
    private_future_years, private_linear_pred, _ = create_forecast(private_totals, forecast_years, 'linear')
    government_future_years, government_linear_pred, _ = create_forecast(government_totals, forecast_years, 'linear')
    
    # Plot historical data
    ax3.plot(years, private_totals.values, marker='o', label='Private (Historical)', linewidth=2)
    ax3.plot(years, government_totals.values, marker='s', label='Government (Historical)', linewidth=2)
    
    # Plot forecasts
    ax3.plot(private_future_years, private_linear_pred, '--', marker='o', 
             label='Private (Linear Forecast)', alpha=0.7, linewidth=2)
    ax3.plot(government_future_years, government_linear_pred, '--', marker='s', 
             label='Government (Linear Forecast)', alpha=0.7, linewidth=2)
    
    ax3.set_title(f'Linear Forecasts ({forecast_years} years ahead)')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Millions of Dollars')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Forecasts - Polynomial
    ax4 = axes[1, 1]
    private_future_years, private_poly_pred, _ = create_forecast(private_totals, forecast_years, 'polynomial')
    government_future_years, government_poly_pred, _ = create_forecast(government_totals, forecast_years, 'polynomial')
    
    # Plot historical data
    ax4.plot(years, private_totals.values, marker='o', label='Private (Historical)', linewidth=2)
    ax4.plot(years, government_totals.values, marker='s', label='Government (Historical)', linewidth=2)
    
    # Plot forecasts
    ax4.plot(private_future_years, private_poly_pred, '--', marker='o', 
             label='Private (Polynomial Forecast)', alpha=0.7, linewidth=2)
    ax4.plot(government_future_years, government_poly_pred, '--', marker='s', 
             label='Government (Polynomial Forecast)', alpha=0.7, linewidth=2)
    
    ax4.set_title(f'Polynomial Forecasts ({forecast_years} years ahead)')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Millions of Dollars')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print forecast summary
    print(f"\n--- {forecast_years}-Year Forecast Summary ---")
    print("Linear Model Predictions:")
    print(f"Private Sector - Final Year: ${private_linear_pred[-1]:,.0f}M")
    print(f"Government - Final Year: ${government_linear_pred[-1]:,.0f}M")
    
    print("\nPolynomial Model Predictions:")
    print(f"Private Sector - Final Year: ${private_poly_pred[-1]:,.0f}M")
    print(f"Government - Final Year: ${government_poly_pred[-1]:,.0f}M")
    
    # Growth analysis
    private_cagr = ((private_totals.iloc[-1] / private_totals.iloc[0]) ** (1/len(years-1)) - 1) * 100
    government_cagr = ((government_totals.iloc[-1] / government_totals.iloc[0]) ** (1/len(years-1)) - 1) * 100
    
    print(f"\nHistorical CAGR:")
    print(f"Private Sector: {private_cagr:.2f}%")
    print(f"Government: {government_cagr:.2f}%")

# --- 6. Execute the analysis ---
try:
    if 'private_totals' in locals() and 'government_totals' in locals():
        plot_with_forecasts(private_totals, government_totals)
    else:
        print("Could not separate private and government data. Please check your data structure.")
        print("Here's what we found in the industry column:")
        print(df.iloc[:, 1].tolist())
        
except Exception as e:
    print(f"An error occurred: {e}")
    print("Let's examine the data structure more carefully...")
    
    # Fallback analysis
    print("\n--- Data Structure Examination ---")
    for i, col in enumerate(df.columns):
        print(f"Column {i}: {col}")
    
    print("\n--- First few rows of data ---")
    print(df.head(10))