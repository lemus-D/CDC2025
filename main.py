import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_name = "Business.xlsx"
sheet_to_load = "Table 1"

# --- 1. Load the data, skipping the top 5 rows and ignoring headers ---
df = pd.read_excel(file_name, sheet_name=sheet_to_load, header=None, skiprows=5)

# --- 2. Manually fix the column headers ---
new_headers = df.iloc[0]
df = df[1:]

# --- THE FIX IS HERE ---
# Manually assign the names for the first two columns.
new_headers.iloc[0] = 'Line'
new_headers.iloc[1] = 'Industry'
# --- END FIX ---

df.columns = new_headers

# --- 3. Clean and convert the data ---
columns_to_convert = df.columns[2:]
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')
df.dropna(how='all', inplace=True)


# --- 4. Check our work ---
print("--- Cleaned DataFrame Info ---")
df.info()

# --- 5. Plot the histogram ---
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x=2023, bins=15)
plt.title("Distribution of Real Value Added by Industry (2023)")
plt.xlabel("Real Value Added (Millions of Dollars)")
plt.ylabel("Number of Industries")
plt.show()


# --- 6. Separate Government vs. Private and Plot ---
# Clean up any extra whitespace from the 'Industry' column
df['Industry'] = df['Industry'].astype(str).str.strip()

# Isolate the 'Federal government' row
government_df = df[df['Industry'] == 'Federal government']

# Isolate all other rows (private sector)
private_df = df[df['Industry'] != 'Federal government']

# Get a list of the year columns to sum up
year_columns = df.columns[2:]

# Sum the contributions for each group
government_sum = government_df[year_columns].sum()
private_sum = private_df[year_columns].sum()

# Create a new DataFrame to hold the summary
summary_df = pd.DataFrame({'Government': government_sum, 'Private Sector': private_sum})

# --- Create the stacked area chart ---
summary_df.plot(
    kind='area',
    stacked=True,
    figsize=(12, 7),
    title='Government vs. Private Sector Contribution to the Space Economy'
)
plt.ylabel("Real Value Added (Millions of Dollars)")
plt.xlabel("Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()