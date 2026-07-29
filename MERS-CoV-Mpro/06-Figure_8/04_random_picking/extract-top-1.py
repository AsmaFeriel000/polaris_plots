import os
import pandas as pd

# Input and output files
input_file = "output/random_pick_all_rmsds.csv"
output_dir = "./output"
output_file = os.path.join(output_dir, "top1_only_random_pick_all_rmsds.csv")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the input CSV
df = pd.read_csv(input_file)

# Keep only rows where N == 1
df_n1 = df[df["N"] == 1]

# Save the filtered data
df_n1.to_csv(output_file, index=False)

print(f"Extracted {len(df_n1)} rows to {output_file}")