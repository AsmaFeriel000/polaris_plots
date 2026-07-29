#!/usr/bin/env python
# coding: utf-8

# # FEgrow: An Open-Source Molecular Builder and Free Energy Preparation Workflow
# 
# **Authors: Mateusz K Bieniek, Ben Cree, Rachael Pirie, Joshua T. Horton, Natalie J. Tatum, Daniel J. Cole**

# ## Overview
# 
# Building and scoring molecules can be further streamlined by employing our established protocol. Here we show how to quickly build a library and score the entire library. 

import os   # afk
from glob import glob  # afk

import pandas as pd
import prody
from rdkit import Chem

import fegrow
from fegrow import ChemSpace

from fegrow.testing import core_5R83_path, rec_5R83_path, data_5R83_path

from dask.distributed import LocalCluster

import os
import shutil

def main():

    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RESULTING_MOLS_DIR = os.path.join(OUTPUT_DIR, "resulting_mols")
    os.makedirs(RESULTING_MOLS_DIR, exist_ok=True)

    BEST_CONFORMERS_DIR = os.path.join(OUTPUT_DIR, "best_conformers")
    os.makedirs(BEST_CONFORMERS_DIR, exist_ok=True)

    lc = LocalCluster(processes=True, n_workers=None, threads_per_worker=1)   

    counter = 1    

    input_folder = "../01-ApoDock-side-chain-modelling/output/complex-mers-lig-h.sdf/receptors_with_hydrogen"        

    # Find all .pdb files in the input folder
    pdb_files = glob(os.path.join(input_folder, "*.pdb"))        

    for pdb_file in pdb_files:    
        # Prepare the ligand template
        print(" pdb file {} read in".format(counter))    

        # scaffold = Chem.SDMolSupplier(core_5R83_path)[0]    
        scaffold = Chem.SDMolSupplier('input/coreh.sdf')[0]

        with open('input/smiles-test-MERS.txt') as f:    
            mols = f.read().splitlines()    

            print("loading core finished round {}".format(counter))
            print("creating chemspace with dask round {}".format(counter))

            # create the chemical space
            cs = ChemSpace(dask_cluster=lc)   
            cs.add_scaffold(scaffold)

            smiles = mols[0:]
            cs.add_smiles(smiles, protonate=True)
            cs

            sys = prody.parsePDB(pdb_file)
            rec = sys.select('not (nucleic or hetatm or water)')
            rec_pdb = os.path.join(OUTPUT_DIR, "rec.pdb")
            rec_final = os.path.join(OUTPUT_DIR, f"rec_final_{counter}.pdb")

            prody.writePDB(rec_pdb, rec)
            fegrow.fix_receptor(rec_pdb, rec_final)
            print("pdb file into rec_final {}".format(counter))

            cs.add_protein(rec_final)
            print("successfully added pdb {} to chemspace to evaluate conformers on it".format(counter))

            cs.evaluate(num_conf=500, gnina_gpu=False, penalty=0.0, al_ignore_penalty=False)

            cs.to_sdf(
                os.path.join(
                    OUTPUT_DIR,
                    f"cs_optimised_molecules_in_rec_{counter}.sdf"
                )
            )

            for i in range(len(cs)):
                pdb_filename = os.path.join(
                    BEST_CONFORMERS_DIR,
                    f"best_conformers_in_rec_{counter}_{i}.pdb"
                )

                sdf_filename = os.path.join(
                    RESULTING_MOLS_DIR,
                    f"rec_{counter}_mol{i}.sdf"
                )

                pdb_first_model = os.path.join(
                    OUTPUT_DIR,
                    f"tmp_first_model_{counter}_{i}.pdb"
                )

                try:
                    # Write the best conformer as a PDB
                    cs[i].to_file(pdb_filename)

                    # Read only the first MODEL (or the whole file if there are no MODEL records)
                    with open(pdb_filename, "r") as infile:
                        lines = infile.readlines()

                    inside_model = False
                    first_model_lines = []

                    for line in lines:
                        if line.startswith("MODEL"):
                            if inside_model:
                                break
                            inside_model = True

                        if inside_model:
                            first_model_lines.append(line)

                        if line.startswith("ENDMDL") and inside_model:
                            break

                    # If there were no MODEL records, keep the whole file
                    if not first_model_lines:
                        first_model_lines = lines

                    with open(pdb_first_model, "w") as outfile:
                        outfile.writelines(first_model_lines)

                    # Convert first model PDB -> SDF
                    os.system(f'obabel -ipdb "{pdb_first_model}" -O "{sdf_filename}"')

                    os.remove(pdb_first_model)

                except AttributeError:
                    print(f"No conformer for molecule {i}")

        cs.df.to_csv(
            os.path.join(OUTPUT_DIR, "MERS-out.csv"),
            index=True
        )

        counter += 1

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Especially needed for frozen executables
    main()
