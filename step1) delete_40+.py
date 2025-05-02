import pandas as pd
import os

# Load the data
#df = pd.read_csv("original data/application_train.csv")
df = pd.read_csv("application_train_merged.csv")

# Calculate the percentage of missing data for each column
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Filter columns with more than 40% missing data
columns_to_keep = missing_percentage[missing_percentage <= 40].index
df_filtered = df[columns_to_keep]

# Create a new folder called "modified data" if it doesn't already exist
os.makedirs("modified data", exist_ok=True)

# Save the filtered dataset to a new file
output_path = "modified data/application_40deleted.csv"
df_filtered.to_csv(output_path, index=False)

print(f"Filtered dataset saved to '{output_path}' with columns having <=40% missing data.")
