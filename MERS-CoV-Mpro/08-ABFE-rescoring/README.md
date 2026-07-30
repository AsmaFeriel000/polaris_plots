## Rescoring the FEgrow poses with absolute binding free energies (ABFE)

The poses built in step 02 are rescored with alchemical absolute binding free energy
calculations, run with [a3fe](https://github.com/michellab/a3fe). For each
molecule, every (molecule, ApoDock receptor) pair listed in the ApoScore results is set up
and run, and the pose belonging to the most favourable free energy is taken as the
prediction. `abfe_analysis.ipynb` then compares those top-scored poses to the released
(unblinded) competition poses.

This was only done for MERS-CoV Mpro.

### Inputs

| What | Where | Used by |
| --- | --- | --- |
| Ligand poses to rescore | `../02-building-poses-in-ApoDock-receptors/output/resulting_mols/` | `run_abfe.py` |
| ApoDock receptors (`rec_final_<rec>.pdb`) | `../02-building-poses-in-ApoDock-receptors/output/` | `run_abfe.py` |
| Original (unmodelled) receptor | `../02-building-poses-in-ApoDock-receptors/output/rec.pdb` | `run_abfe.py` |
| (molecule, receptor) pairs | `../06-Figure_8/02_ApoScore/output/mol_scores_sorted.csv` | both |
| Rescored poses | `input/poses/` | notebook |
| True poses | `../../released-receptor-structures/released_test_molecules/test_mers.sdf` | notebook |
| Lowest-RMSD cross-check | `../06-Figure_8/01_lowest_rmsd/output/lowest_rmsds.csv` | notebook |

`input/poses/` holds the 401 poses that were rescored. They are identical to their
counterparts under step 02, but the notebook reads this local copy so that it keeps
reproducing the published numbers even if the step 02 poses are regenerated.

### Outputs

Everything is written to `output/`:

* `output/abfe/mol*/rec_final_*/` - the a3fe calculation tree. This is large and is **not** committed.
* `output/abfe_df_processed.csv` - the fully-processed per-(molecule, receptor) dataframe
  (free energies, pose RMSDs, equilibrated-pose RMSDs). This **is** committed, so the
  analysis notebook reproduces without the raw calculation tree.
* `output/rmsd_cdf_plot.png`, `output/abfe_complete_results.csv`,
  `output/abfe_best_conformations.csv`, `output/abfe_summary_stats.txt` - written by the
  notebook.

## Running the calculations (`run_abfe.py`)

Install `a3fe` **0.4.0** (which brings in BioSimSpace and SOMD) following its
[installation instructions](https://github.com/michellab/a3fe), then add the extras used by
this script and the analysis notebook:

```
mamba install -c conda-forge -c openeye typer loguru pymol-open-source openeye-toolkits spyrmsd
```

The results committed here were produced with a3fe 0.4.0.

Note that the H atom of the N-terminal SER residues in the input PDB files has to be renamed
by hand before parameterisation:

```
Line 11 H -> H1
```

Run a given molecule (or several) with:

```
python run_abfe.py mol3 mol4
```

With no arguments every `mol_id` in `mol_scores_sorted.csv` is run. The script submits
through SLURM: the `SlurmConfig` in `run_abfe.py` (partition, account, GPU resources) is
specific to the cluster this was run on and will need editing elsewhere.

## Running the analysis (`abfe_analysis.ipynb`)

The notebook runs from the committed `output/abfe_df_processed.csv` and the committed poses,
so no raw calculation data is needed. Use the environment from the plotting directory:

```
conda env create -f ../../plotting/polaris-env.yaml
conda activate polaris-env
jupyter lab abfe_analysis.ipynb
```

Start Jupyter from this directory, as the notebook resolves its paths relative to the
working directory. (`openff-toolkit` is imported lazily by the optional 3D-viewer helper
only; the rest of the notebook does not need it.)

Two sections need the raw a3fe tree: the equilibrated-pose RMSDs (section 13) and the pose
PDB export (section 14). They skip themselves when it is absent. Point the notebook at a
copy of the tree with

```
ABFE_RUN_DIR=/path/to/abfe jupyter lab abfe_analysis.ipynb
```

Set `FORCE_REBUILD = True` in the configuration cell (or delete
`output/abfe_df_processed.csv`) to rebuild the cache from the raw data.
