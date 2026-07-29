import os
import subprocess
import re

"""Build protein-ligand complexes by merging receptor PDBs with ligand SDFs.
Assumes receptor files are named "rec_final_<ID>.pdb" and ligand files are named
"rec_<ID>_mol<MOLID>.sdf", where <ID> matches between receptor and ligand.
The output complex PDBs are saved as "rec_<ID>_mol<MOLID>.pdb".
Requires Open Babel (obabel) to convert SDF to PDB format.

"""

BASE_DIR = "../02-building-poses-in-ApoDock-receptors/output"
LIG_DIR = os.path.join(BASE_DIR, "resulting_mols")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(SCRIPT_DIR, "output")
TMP_DIR = os.path.join(OUT_DIR, "tmp_lig")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

rec_pattern = re.compile(r"rec_final_(\d+)\.pdb")
lig_pattern = re.compile(r"rec_(\d+)_mol(\d+)\.sdf")

print(os.listdir(BASE_DIR))
print(os.listdir(LIG_DIR))

# map receptors
receptors = {}
for f in os.listdir(BASE_DIR):
    m = rec_pattern.match(f)

    if m:
        receptors[m.group(1)] = os.path.join(BASE_DIR, f)

for lig_file in os.listdir(LIG_DIR):
    
    m = lig_pattern.match(lig_file)
    if not m:
        continue

    rec_id, mol_id = m.groups()
    if rec_id not in receptors:
        print(f" Missing receptor for {lig_file}")
        continue

    receptor_pdb = receptors[rec_id]
    ligand_sdf = os.path.join(LIG_DIR, lig_file)

    ligand_pdb = os.path.join(TMP_DIR, f"lig_{rec_id}_{mol_id}.pdb")
    output_pdb = os.path.join(
        OUT_DIR, f"rec_{rec_id}_mol{mol_id}.pdb"
    )

    # 1. Convert ligand SDF → PDB
    subprocess.run([
        "obabel",
        ligand_sdf,
        "-O", ligand_pdb,
        "--resname", "LIG"
    ], check=True)

    # 2. Merge receptor + ligand by concatenation
    with open(output_pdb, "w") as out:
        with open(receptor_pdb) as rec:
            for line in rec:
                if not line.startswith("END"):
                    out.write(line)

        with open(ligand_pdb) as lig:
            for line in lig:
                if line.startswith(("ATOM", "HETATM")):
                    out.write(line)

        out.write("END\n")

    print(f"Created {os.path.basename(output_pdb)}")

print("All complexes built correctly.")