from dask.distributed import Client
from dask.distributed import LocalCluster

from rdkit import Chem
from rdkit.Chem import rdFMCS

import pathlib
import traceback
import tqdm
import dask
import fegrow
import time


# ============================================================
# DATA CONTAINER
# ============================================================

class ReferenceComplex:
    def __init__(
        self,
        ligand_name,
        ligand_mol,
        receptor_file,
    ):
        self.ligand_name = ligand_name
        self.ligand_mol = ligand_mol
        self.receptor_file = receptor_file


# ============================================================
# CREATE RECEPTOR-ONLY FILES
# ============================================================

def prepare_receptors():

    complex_dir = pathlib.Path("complex-train")
    receptor_dir = pathlib.Path("receptors")

    receptor_dir.mkdir(exist_ok=True)

    complex_files = list(complex_dir.glob("*.pdb"))

    print(f"Preparing {len(complex_files)} receptors")

    for pdb_file in complex_files:

        output_lines = []

        with open(pdb_file) as f:

            for line in f:

                # Keep ONLY protein atoms
                if line.startswith("ATOM"):
                    output_lines.append(line)

        receptor_name = pdb_file.name.replace(
            "complex",
            "receptor"
        )

        out_file = receptor_dir / receptor_name

        with open(out_file, "w") as f:
            f.writelines(output_lines)

    print("Finished preparing receptors")


# ============================================================
# MCS MATCHING
# ============================================================

def find_best_n_matches(
    target_ligand,
    ref_complexes,
    n_top=10,
):

    matched_mols = []

    for ref_obj in ref_complexes:

        ref_lig = ref_obj.ligand_mol

        try:

            mcs = rdFMCS.FindMCS(
                [target_ligand, ref_lig],
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
                atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                bondCompare=rdFMCS.BondCompare.CompareAny,
                maximizeBonds=False,
                timeout=1,
            )

            matched_mols.append(
                (
                    mcs.numAtoms,
                    mcs.numBonds,
                    ref_obj,
                    mcs.smartsString,
                )
            )

        except Exception:
            continue

    matched_mols.sort(
        key=lambda x: (x[0], x[1]),
        reverse=True,
    )

    return matched_mols[:n_top]


# ============================================================
# COORDINATE TRANSFER
# ============================================================

def transfer_coordinates(
    reference_ligand: Chem.Mol,
    template_ligand: Chem.Mol,
) -> Chem.Mol:

    matches = reference_ligand.GetSubstructMatch(
        template_ligand
    )

    if not matches:
        raise RuntimeError(
            "Could not map template ligand"
        )

    ref_conformer = reference_ligand.GetConformer(0)

    template_conformer = Chem.Conformer(
        template_ligand.GetNumAtoms()
    )

    for i, atom_match in enumerate(matches):

        pos = ref_conformer.GetAtomPosition(atom_match)

        template_conformer.SetAtomPosition(i, pos)

    template_ligand.AddConformer(
        template_conformer,
        assignId=True,
    )

    return template_ligand


# ============================================================
# MAIN POSE GENERATION
# ============================================================

@dask.delayed
def pose_ligand(
    target_ligand,
    reference_complexes,
    ligand_name,
):

    best_matches = find_best_n_matches(
        target_ligand=target_ligand,
        ref_complexes=reference_complexes,
        n_top=10,
    )

    if len(best_matches) == 0:

        return {
            "failed": True,
            "name": ligand_name,
            "reason": "No MCS matches",
        }

    successful_pose = False

    for close_match in best_matches:

        ref_obj = close_match[2]

        reference_ligand = ref_obj.ligand_mol
        receptor_file = ref_obj.receptor_file
        template_name = ref_obj.ligand_name

        print("=" * 80)
        print(f"Target ligand: {ligand_name}")
        print(f"Template ligand: {template_name}")
        print(f"Receptor: {receptor_file}")

        try:

            # ------------------------------------------------
            # Build MCS
            # ------------------------------------------------

            mcs_smarts = close_match[-1]

            mcs_mol = Chem.MolFromSmarts(
                mcs_smarts
            )

            if mcs_mol is None:
                continue

            # ------------------------------------------------
            # Transfer coordinates
            # ------------------------------------------------

            mcs_mol = transfer_coordinates(
                reference_ligand=reference_ligand,
                template_ligand=mcs_mol,
            )

            # ------------------------------------------------
            # Create FEgrow molecule
            # ------------------------------------------------

            fe_mol = fegrow.RMol(target_ligand)

            # ------------------------------------------------
            # Save template
            # ------------------------------------------------

            fe_mol._save_template(mcs_mol)

            # ------------------------------------------------
            # Generate conformers
            # ------------------------------------------------

            try:

                fe_mol.generate_conformers(
                    num_conf=300
                )

            except Exception:

                print(
                    "Conformer generation failed"
                )

                continue

            n_conf = fe_mol.GetNumConformers()

            print(
                f"Generated conformers: {n_conf}"
            )

            if n_conf == 0:
                continue

            # ------------------------------------------------
            # Remove clashes
            # ------------------------------------------------

            try:

                fe_mol.remove_clashing_confs(
                    protein=receptor_file
                )

            except Exception:

                print(
                    "Clash removal failed"
                )

                continue

            n_conf = fe_mol.GetNumConformers()

            print(
                f"After clash removal: {n_conf}"
            )

            if n_conf == 0:

                print(
                    "All conformers removed"
                )

                continue

            # ------------------------------------------------
            # Optimize in receptor
            # ------------------------------------------------

            try:

                fe_mol.optimise_in_receptor(
                    receptor_file=receptor_file,
                    ligand_force_field="openff",
                )

            except Exception:

                print(
                    "Optimization failed"
                )

                continue

            n_conf = fe_mol.GetNumConformers()

            print(
                f"After optimization: {n_conf}"
            )

            if n_conf == 0:
                continue

            # ------------------------------------------------
            # Sort conformers
            # ------------------------------------------------

            fe_mol.sort_conformers()

            successful_pose = True

            break

        except Exception:

            traceback.print_exc()

            continue

    # ========================================================
    # HANDLE FAILURES
    # ========================================================

    if not successful_pose:

        return {
            "failed": True,
            "name": ligand_name,
            "reason": "No successful template",
        }

    # Keep only lowest energy conformer
    fe_mol = fegrow.RMol(
        fe_mol,
        confId=0,
    )

    return {
        "failed": False,
        "posed_mol": fe_mol,
        "template_name": template_name,
        "template_mol": fegrow.RMol(
            reference_ligand
        ),
        "receptor_file": receptor_file,
        "name": ligand_name,
    }


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_reference_complexes():

    ligand_dir = pathlib.Path("ligand-train")
    receptor_dir = pathlib.Path("receptors")

    reference_complexes = []

    ligand_files = sorted(
        ligand_dir.glob("*.sdf")
    )

    print(
        f"Found {len(ligand_files)} ligand files"
    )

    for lig_file in ligand_files:

        ligand_name = lig_file.stem

        # --------------------------------------------
        # mol-0 -> receptor-0.pdb
        # --------------------------------------------

        try:

            lig_index = ligand_name.split("-")[-1]

        except Exception:

            print(
                f"Could not parse {ligand_name}"
            )

            continue

        receptor_file = (
            receptor_dir /
            f"receptor-{lig_index}.pdb"
        )

        if not receptor_file.exists():

            print(
                f"Skipping {ligand_name}: "
                f"{receptor_file.name} missing"
            )

            continue

        try:

            rdkit_mol = Chem.MolFromMolFile(
                lig_file.as_posix(),
                removeHs=True,
            )

            if rdkit_mol is None:

                print(
                    f"Failed loading {ligand_name}"
                )

                continue

            if rdkit_mol.GetNumConformers() == 0:

                print(
                    f"{ligand_name} has no coords"
                )

                continue

            reference_complexes.append(
                ReferenceComplex(
                    ligand_name=ligand_name,
                    ligand_mol=rdkit_mol,
                    receptor_file=str(
                        receptor_file.absolute()
                    ),
                )
            )

        except Exception:

            traceback.print_exc()

            continue

    print(
        f"Loaded {len(reference_complexes)} "
        f"reference complexes"
    )

    return reference_complexes


# ============================================================
# LOAD TEST MOLECULES
# ============================================================

def load_test_molecules(
    smiles_file="smiles-test-MERS.txt"
):

    test_mols = []

    supplier = Chem.SmilesMolSupplier(
        smiles_file
    )

    for i, lig in enumerate(supplier):

        if lig is None:

            print(
                f"Failed parsing ligand {i}"
            )

            continue

        lig = Chem.AddHs(lig)

        test_mols.append(lig)

    print(
        f"Loaded {len(test_mols)} test ligands"
    )

    return test_mols


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1: PREPARE RECEPTORS
    # --------------------------------------------------------

    prepare_receptors()

    # --------------------------------------------------------
    # STEP 2: START DASK
    # --------------------------------------------------------

    workers = 20

    client = Client(
        LocalCluster(
            threads_per_worker=1,
            n_workers=workers,
        )
    )

    print(f"Dask client created: {client}")

    # --------------------------------------------------------
    # STEP 3: LOAD REFERENCES
    # --------------------------------------------------------

    reference_complexes = (
        load_reference_complexes()
    )

    if len(reference_complexes) == 0:

        raise RuntimeError(
            "No valid reference complexes found"
        )

    # --------------------------------------------------------
    # STEP 4: LOAD TEST LIGANDS
    # --------------------------------------------------------

    test_mols = load_test_molecules()

    # --------------------------------------------------------
    # STEP 5: SHARE REFERENCES
    # --------------------------------------------------------

    delayed_refs = dask.delayed(
        reference_complexes
    )

    # --------------------------------------------------------
    # STEP 6: BUILD TASKS
    # --------------------------------------------------------

    tasks = [

        pose_ligand(
            target_ligand=test_lig,
            reference_complexes=delayed_refs,
            ligand_name=f"ligand_{i}",
        )

        for i, test_lig in enumerate(test_mols)
    ]

    # --------------------------------------------------------
    # STEP 7: SUBMIT TASKS
    # --------------------------------------------------------

    submitted = client.compute(tasks)

    # --------------------------------------------------------
    # STEP 8: OUTPUTS
    # --------------------------------------------------------

    output_path = pathlib.Path("outputs")
    output_path.mkdir(exist_ok=True)

    failed_ligands = []

    # --------------------------------------------------------
    # STEP 9: MONITOR
    # --------------------------------------------------------

    with tqdm.tqdm(
        total=len(submitted),
        desc="Docking ligands",
        ncols=100,
    ) as pbar:

        while len(submitted) > 0:

            completed_jobs = []

            for job in submitted:

                if not job.done():
                    continue

                completed_jobs.append(job)

                pbar.update(1)

                try:

                    result = job.result()

                    # ------------------------------------
                    # Failed ligand
                    # ------------------------------------

                    if result.get("failed", False):

                        failed_ligands.append(
                            result["name"]
                        )

                        print(
                            f"FAILED: {result['name']}"
                        )

                        continue

                    ligand_name = result["name"]

                    template_name = (
                        result["template_name"]
                    )

                    receptor_file = (
                        result["receptor_file"]
                    )

                    out_dir = (
                        output_path /
                        ligand_name
                    )

                    out_dir.mkdir(
                        exist_ok=True,
                        parents=True,
                    )

                    posed_ligand = (
                        result["posed_mol"]
                    )

                    template_lig = (
                        result["template_mol"]
                    )

                    # ------------------------------------
                    # Save best pose
                    # ------------------------------------

                    posed_ligand.to_file(
                        str(
                            out_dir /
                            "best_pose.sdf"
                        )
                    )

                    # ------------------------------------
                    # Save template
                    # ------------------------------------

                    template_lig.to_file(
                        str(
                            out_dir /
                            "template_lig.sdf"
                        )
                    )

                    # ------------------------------------
                    # Save metadata
                    # ------------------------------------

                    with open(
                        out_dir / "metadata.txt",
                        "w",
                    ) as f:

                        f.write(
                            f"Target ligand: "
                            f"{ligand_name}\n"
                        )

                        f.write(
                            f"Template ligand: "
                            f"{template_name}\n"
                        )

                        f.write(
                            f"Receptor: "
                            f"{receptor_file}\n"
                        )

                except Exception:

                    traceback.print_exc()

            # Remove completed jobs
            for job in completed_jobs:
                submitted.remove(job)

            time.sleep(5)

    # --------------------------------------------------------
    # STEP 10: SAVE FAILURES
    # --------------------------------------------------------

    with open(
        "failed_ligands.txt",
        "w",
    ) as f:

        for lig in failed_ligands:
            f.write(f"{lig}\n")

    print(
        f"Failed ligands: "
        f"{len(failed_ligands)}"
    )

    print("All ligands processed")

    client.close()


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()