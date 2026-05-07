#import csv

#count = 0
#total = 0

#with open("complex_rmsd_results-submitted-vs-ref-5-residues.csv") as f:
#    reader = csv.reader(f)
#    next(reader)  # skip header
#    for row in reader:
#        rmsd = float(row[1])
#        total += 1
#        if rmsd < 2.0:
#            count += 1

#percent = 100 * count / total if total else 0
#print(f"{percent:.2f}% of structures have RMSD < 2 Å")

import pandas as pd

df = pd.read_csv("complex_rmsd_results-submitted-vs-ref-5-residues.csv")

total = len(df)
passed = (df["pocket_RMSD_A"] < 2.0).sum()

percentage = (passed / total) * 100

print(f"Total structures: {total}")
print(f"Pocket RMSD < 2 Å: {passed}")
print(f"Percentage: {percentage:.2f}%")
