from pathlib import Path
import pandas as pd
import shutil
import typer
import loguru
import BioSimSpace as BSS
import a3fe as a3
from a3fe.read._process_bss_systems import rename_lig
import subprocess
from BioSimSpace.Sandpit.Exscientia._SireWrappers import Molecule as _Molecule
from sire.legacy import Mol as _SireMol
from pymol import cmd


logger = loguru.logger


# All input paths are resolved relative to this script, so it can be run from any
# working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
# FEgrow poses and ApoDock receptors from step 02
POSE_DIR = (SCRIPT_DIR / ".." / "02-building-poses-in-ApoDock-receptors" / "output").resolve()
LIGAND_DIR = POSE_DIR / "resulting_mols"  # rec_<rec>_mol<mol>.sdf
RECEPTOR_DIR = POSE_DIR  # rec_final_<rec>.pdb
ORIGINAL_RECPT = POSE_DIR / "rec.pdb"
# ApoScore ranking of (molecule, receptor) pairs from step 06
RESULTS_CSV = (
    SCRIPT_DIR / ".." / "06-Figure_8" / "02_ApoScore" / "output" / "mol_scores_sorted.csv"
).resolve()
OUTPUT_DIR = SCRIPT_DIR / "output" / "abfe"
#A3FE_TEMPLATE_FILE_DIR = Path("a3fe_template_files").absolute()
LIGAND_FORCE_FIELD = "openff_unconstrained-2.2.1"
PROTEIN_FORCE_FIELD = "ff14SB"
POCKET_RESIDUES: list[str] = [
    'MET25 A', 'THR26 A', 'LEU27 A', 'HIS41 A', 'LEU49 A',
    'PHE143 A', 'LEU144 A', 'CYS145 A', 'GLY146 A', 'SER147 A',
    'CYS148 A', 'HIS166 A', 'GLN167 A', 'MET168 A', 'GLU169 A',
    'LEU170 A', 'ALA171 A', 'HIS175 A', 'MET184 A', 'MET189 A',
    'ASP190 A', 'LYS191 A', 'GLN192 A', 'VAL193 A', 'SER1 B'
]
    

app = typer.Typer()


def sdf_to_prm7(sdf_path: Path, output_path: Path) -> None:
    """Convert an SDF file to a PRM7 file using BioSimSpace."""
    logger.info(f"Parameterising {sdf_path} to {output_path} using {LIGAND_FORCE_FIELD}")
    mol = BSS.IO.readMolecules(str(sdf_path))[0]
    param_mol = BSS.Parameters.parameterise(molecule=mol, forcefield=LIGAND_FORCE_FIELD).getMolecule()
    BSS.IO.saveMolecules(str(output_path.with_suffix("")), param_mol, "prm7")

def sdf_to_rst7(sdf_path: Path, output_path: Path) -> None:
    """Convert an SDF file to a RST7 file using BioSimSpace."""
    logger.info(f"Converting {sdf_path} to {output_path}")
    mol = BSS.IO.readMolecules(str(sdf_path))[0]
    BSS.IO.saveMolecules(str(output_path.with_suffix("")), mol, "rst7")


def merge_pdb_with_pocket(main_pdb: str, pocket_pdb: str, output_pdb: str, pocket_residues: list[str] = POCKET_RESIDUES) -> None:
    """
    Merge two PDB structures, using pocket residues from one structure
    and the rest from another.
    
    Parameters
    ----------
    main_pdb : str
        Path to the main PDB structure
    pocket_pdb : str
        Path to the PDB containing pocket residues to use
    output_pdb : str
        Path where the merged structure will be saved
    """
    
    cmd.delete('all')
    
    cmd.load(main_pdb, 'main_struct')
    cmd.load(pocket_pdb, 'pocket_struct')
    
    pocket_selection = ' or '.join([
        f'(resn {res.split()[0][0:3]} and resi {res.split()[0][3:]} and chain {res.split()[1]})'
        for res in pocket_residues
    ])
    
    cmd.remove(f'main_struct and ({pocket_selection})')
    cmd.create('pocket_obj', f'pocket_struct and ({pocket_selection})')
    cmd.create('merged', 'main_struct or pocket_obj')
    
    cmd.save(output_pdb, 'merged')
    cmd.delete('all')
    
    logger.info(f"Merged structure saved to: {output_pdb}")

def pdb_to_rst7_and_prm7(pdb_path: Path, rst7_path: Path, prm7_path: Path) -> None:
    """Convert a PDB file to a RST7 and PRM7 file using tleap"""
    logger.info(f"Converting {pdb_path} to {rst7_path} and {prm7_path}")

    # First, process with pdb4amber to fix common issues
    fixed_pdb_path = pdb_path.with_name(pdb_path.stem + "_fixed.pdb")
    subprocess.run(["pdb4amber", "-i", str(pdb_path), "-o", str(fixed_pdb_path)], check=True)

    # Next, manually rename the Hs in col 3 with atom numbers 4548 anmd 4547 to H1 (from H)
    lines = fixed_pdb_path.read_text().splitlines()
    lines_to_write = []
    for line in lines:
        split_line = line.split()
        if not len(split_line) > 2:
            lines_to_write.append(line)
        elif split_line[2] == "H" and split_line[5] in {"1", "302"}:
            lines_to_write.append(line[:12] + " H1" + line[15:])
        else:
            lines_to_write.append(line)
    fixed_pdb_path.write_text("\n".join(lines_to_write) + "\n")

    # Now, parameterise with tleap
    script_path = pdb_path.parent / "tleap_script.in"
    tleap_script = f"""
    source leaprc.protein.ff14SB
    protein = loadpdb {fixed_pdb_path}
    saveamberparm protein {rst7_path} {prm7_path}
    quit
    """
    script_path.write_text(tleap_script)
    subprocess.run(["tleap", "-f", str(script_path)], check=True)
    script_path.unlink()

def rename_lig(
    bss_system: BSS._SireWrappers._system.System, new_name: str = "LIG"
) -> None:  # type: ignore
    """Rename the ligand in a BSS system.

    Parameters
    ----------
    bss_system : BioSimSpace.Sandpit.Exscientia._SireWrappers._system.System
        The BSS system.
    new_name : str
        The new name for the ligand.
    Returns
    -------
    None
    """
    # Ensure that we only have one molecule
    if len(bss_system) != 1:
        raise ValueError("BSS system must only contain one molecule.")

    # Extract the sire object for the single molecule
    mol = bss_system[0]
    mol_sire = mol._sire_object

    # Create an editable version of the sire object
    mol_edit = mol_sire.edit()

    # Rename the molecule and the residue to the supplied name
    resname = _SireMol.ResName(new_name)  # type: ignore
    mol_edit = mol_edit.residue(_SireMol.ResIdx(0)).rename(resname).molecule()  # type: ignore
    mol_edit = mol_edit.edit().rename(new_name).molecule()

    # Commit the changes and update the system
    mol._sire_object = mol_edit.commit()
    bss_system.updateMolecule(0, mol)

def assemble_param_systems(base_dir: Path, receptor_rst7_path: Path, receptor_prm7_path: Path, ligand_prm7_path: Path, ligand_rst7_path: Path) -> None:
    """Assemble the parameterised input into the files needed for a3fe."""
    ligand_sys = BSS.IO.readMolecules([str(ligand_rst7_path), str(ligand_prm7_path)])
    rename_lig(ligand_sys, "LIG")
    ligand = ligand_sys[0]
    receptor = BSS.IO.readMolecules([str(receptor_rst7_path), str(receptor_prm7_path)]).getMolecules()
    complex = ligand + receptor

    # Save the required output files
    BSS.IO.saveMolecules(str(base_dir / "free_param"), ligand, ["prm7", "rst7"])
    BSS.IO.saveMolecules(str(base_dir / "bound_param"), complex, ["prm7", "rst7"])


def write_inputs_mol_and_receptor(ligand: str, receptor: str, parent_dir: Path, lig_prm7_path: Path) -> None:
    """Write the input files for a given mol_id and receptor to the parent_dir.

    Args:
        ligand (str): The ligand name (with suffix).
        receptor (str): The receptor name (with suffix).
        parent_dir (Path): The parent directory to write the input files to.
        lig_prm7_path (Path): The path to the parameterised ligand file.
    """
    # Remove the .pdb suffix from the names
    receptor_no_suffix = receptor.replace(".pdb", "")

    # Create the output directory if it doesn't exist
    # Note that the mol name does not provide any more information, so we do not use it
    # to name the directory
    calc_dir = parent_dir / receptor_no_suffix
    if calc_dir.exists():
        logger.warning(f"Directory {calc_dir} already exists. Skipping...")
        return

    input_files_dir = calc_dir / "input"
    input_files_dir.mkdir(parents=True, exist_ok=False)

    # Populate the input directory with the necessary files
    shutil.copy(lig_prm7_path, input_files_dir / "ligand.prm7")
    sdf_to_rst7(LIGAND_DIR / ligand, input_files_dir / "ligand.rst7")
    protein_prm7_path = input_files_dir / "receptor.prm7"
    protein_rst7_path = input_files_dir / "receptor.rst7"

    # Create the merged PDB file. This is written into the calculation directory so that
    # the (committed) step 02 output directory is never modified.
    merged_pdb_path = input_files_dir / f"{receptor_no_suffix}_pocket_only.pdb"
    merge_pdb_with_pocket(str(ORIGINAL_RECPT), str(RECEPTOR_DIR / receptor), str(merged_pdb_path))

    pdb_to_rst7_and_prm7(merged_pdb_path, protein_rst7_path, protein_prm7_path)
#    for file in A3FE_TEMPLATE_FILE_DIR.iterdir():
#        shutil.copy(file, input_files_dir / file.name)

    assemble_param_systems(
        base_dir=input_files_dir,
        receptor_rst7_path=input_files_dir / "receptor.rst7",
        receptor_prm7_path=input_files_dir / "receptor.prm7",
        ligand_prm7_path=input_files_dir / "ligand.prm7",
        ligand_rst7_path=input_files_dir / "ligand.rst7",
    )

def write_inputs(mol_id: str, df: pd.DataFrame, out_dir: Path) -> None:
    """Write the input files for a given mol_id to the out_dir.
    
    Args:
        mol_id (str): The molecule ID to write the input files for.
        df (pd.DataFrame): The dataframe containing the results.
        out_dir (Path): The output directory to write the input files to.
    """

    # Create output directory for this molecule
    mol_dir = out_dir / mol_id
    if mol_dir.exists():
        logger.warning(f"Directory {mol_dir} already exists. Skipping...")
        return

    mol_dir.mkdir(parents=True)

    # Copy the relevant rows to a new CSV file
    mol_df = df[df["mol_id"] == mol_id]
    mol_df.to_csv(mol_dir / "mol_scores_sorted.csv", index=False)

    # Parameterise the ligand (so that we only do this once) and
    # save
    ligand_path = LIGAND_DIR / mol_df.iloc[0]["ligand"]
    ligand_output_path = mol_dir / "ligand.prm7"
    sdf_to_prm7(ligand_path, ligand_output_path)

    # Write out the input files for each receptor
    for row in mol_df.itertuples():
        write_inputs_mol_and_receptor(
            ligand=row.ligand, receptor=row.receptor, parent_dir=mol_dir, lig_prm7_path=ligand_output_path
        )

def run_abfe(mol_id: str, out_dir: Path) -> None:
    """Run ABFE using a3fe for the given mol_id"""

    mol_dir = out_dir / mol_id
    # Ensemble size is 5 by default
    somd_cfg = a3.SomdSystemPreparationConfig(runtime_npt_unrestrained=50, # ps
                                         runtime_npt=50, # ps
                                         ensemble_equilibration_time=100,
                                         steps=90_000) # Minimisation steps -- need lots in case of clashes
    slurm_config = a3.SlurmConfig(partition = "gpu-s_paid", time = "1:00:00", gres="gpu:1", extra_options={"mem":"30G", "cpus-per-gpu":"30", "gpus":"1", "account":"comettestgroup3"}, queue_check_interval=10,job_submission_wait=30)
    calc_set = a3.CalcSet(calc_paths=[str(mol_dir / receptor) for receptor in mol_dir.iterdir() if receptor.is_dir()],
                          base_dir=mol_dir,slurm_config=slurm_config)


    calc_set.setup(sysprep_config=somd_cfg)

    # Max job submission changed to 250...
    for calc in calc_set.calcs:
        for leg in calc.legs:
            for stage in leg.stages:
                stage.virtual_queue.queue_len_lim = 200

        if not round(calc.tot_simtime, 3) == 26.000:
            calc.clean()
            calc.run(adaptive=False, runtime=0.1)
            calc.wait()
            calc.set_equilibration_time(0.02)

    # calc_set.clean()
    # calc_set.run(adaptive=False, runtime=0.1) # ns
    # calc_set.wait()
    calc_set.set_equilibration_time(0.02)
    calc_set.analyse(compare_to_exp=False)
    calc_set.save()

@app.command()
def run(mol_ids: list[str] | None = None) -> None:
    """
    Run ABFE calculations for the given mol_ids. If no 
    mol_ids are provided, run for all mol_ids in the results CSV.
    """

    # Get the results dataframe
    results = pd.read_csv(RESULTS_CSV)
    # Remove the "-noH" suffix from the receptor column
    results["receptor"] = results["receptor"].str.replace("-noH", "", regex=False)
    results.head()

    # Get the mol_ids to run
    valid_mol_ids = results["mol_id"].unique().tolist()
    if mol_ids is None:
        mol_ids = valid_mol_ids
    else:
        for mol_id in mol_ids:
            if mol_id not in valid_mol_ids:
                raise ValueError(f"Invalid mol_id: {mol_id}")

    # Run each mol_id
    for mol_id in mol_ids:
        logger.info(f"Running mol_id: {mol_id}")
        write_inputs(mol_id, results, OUTPUT_DIR)
        run_abfe(mol_id, OUTPUT_DIR)


if __name__ == "__main__":
    app()
