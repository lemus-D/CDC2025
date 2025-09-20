import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_name = "Business.xlsx"
sheet_to_load = "Table 1"

# --- 1. Load the data, intentionally skipping the top 5 rows and ignoring headers ---
# This brings in the raw data without letting pandas make wrong guesses.
df = pd.read_excel(file_name, sheet_name=sheet_to_load, header=None, skiprows=5)

# --- 2. Manually fix the column headers ---
# The first row (index 0) of our new DataFrame contains the correct column names.
# Let's grab them.
new_headers = df.iloc[0] 
# Now, let's remove that row from the DataFrame, so only data remains.
df = df[1:] 
# Finally, let's set the column names we just grabbed.
df.columns = new_headers

# --- 3. Clean and convert the data ---
# Get a list of all columns that we expect to be numeric (the years).
# The first two columns are text ('Line' and 'Industry'), so we skip them.
columns_to_convert = df.columns[2:]

# Loop through those columns and force them to be numeric.
# errors='coerce' will turn any problematic values (like text or symbols) into empty 'NaN' values.
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Drop any rows that are now completely empty, which cleans up the bottom of the file.
df.dropna(how='all', inplace=True)


# --- 4. Check our work ---
print("--- Cleaned DataFrame Info ---")
df.info() # You should now see float64 or int64 for the year columns!

print("\n--- Cleaned First 5 Rows ---")
print(df.head())


# --- 5. Plot the clean data ---
plt.figure(figsize=(10, 6))
# The column name '2023' might be a number (2023) or text ('2023'). Let's try it as a number first.
sns.histplot(data=df, x=2023, bins=15)
plt.title("Distribution of Real Value Added by Industry (2023)")
plt.xlabel("Real Value Added (Millions of Dollars)")
plt.ylabel("Number of Industries")
plt.show()