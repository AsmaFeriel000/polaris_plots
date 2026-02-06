import csv

count = 0
total = 0

with open("lowest_rmsds.csv") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        rmsd = float(row[1])
        total += 1
        if rmsd < 2.0:
            count += 1

percent = 100 * count / total if total else 0
print(f"{percent:.2f}% of structures have RMSD < 2 Å")