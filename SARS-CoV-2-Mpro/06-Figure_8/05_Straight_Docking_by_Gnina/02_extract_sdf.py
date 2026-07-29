input_file = "docked.sdf"
output_file = "output_every_9th.sdf"

with open(input_file, "r") as f:
    content = f.read().strip()

# Split into individual molecules
molecules = content.split("$$$$")
molecules = [m.strip() for m in molecules if m.strip()]

# Select every 9th molecule starting from the first (index 0)
selected = [molecules[i] for i in range(0, len(molecules), 9)]

# Write to a new SDF
with open(output_file, "w") as f:
    for mol in selected:
        f.write(mol.strip() + "\n$$$$\n\n\n\n")

print(f"Extracted {len(selected)} molecules into {output_file}")
