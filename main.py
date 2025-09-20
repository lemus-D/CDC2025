import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_name = "Business.xlsx"
sheet_to_load = "Table 1"

# --- 1. Load the data, intentionally skipping the top 5 rows and ignoring headers ---
df = pd.read_excel(file_name, sheet_name=sheet_to_load, header=None, skiprows=5)

# --- 2. Manually fix the column headers ---
new_headers = df.iloc[0] 
df = df[1:] 
df.columns = new_headers

# --- 3. Clean and convert the data ---
columns_to_convert = df.columns[2:]
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')
df.dropna(how='all', inplace=True)


# --- 4. Check our work ---
print("--- Cleaned DataFrame Info ---")
df.info()

print("\n--- Cleaned First 5 Rows ---")
print(df.head())


# --- 5. Plot the changes over the years ---
# The industry column is the second column (at index 1)
# Use .iloc[:, 1] to select the second column by its integer position
industry_series = df.iloc[:, 1]

# Find start and end indices for private and government sectors
# Use the .astype(str) to ensure that the column is treated as a string type
private_start = df[industry_series.astype(str).str.contains('Private industries', case=False, na=False)].index[0]
government_start = df[industry_series.astype(str).str.contains('Government', case=False, na=False)].index[0]

# Sum the values for each year
private_totals = df.loc[private_start+1:government_start-1, columns_to_convert].sum()
government_totals = df.loc[government_start+1:, columns_to_convert].sum()

# --- 6. Plot the changes over the years ---
plt.figure(figsize=(10, 6))

plt.plot(private_totals.index, private_totals.values, marker='o', label='Private Sector', color='#1f77b4')
plt.plot(government_totals.index, government_totals.values, marker='o', label='Government', color='#ff7f0e')

plt.title("Real Value Added by Sector Over the Years")
plt.xlabel("Year")
plt.ylabel("Millions of Dollars")
plt.legend()
plt.grid(True)
plt.show()
