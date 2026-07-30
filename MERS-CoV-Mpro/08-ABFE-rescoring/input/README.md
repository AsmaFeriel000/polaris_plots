# The poses that were rescored

`poses/` holds the 401 FEgrow poses (one per rescored (molecule, receptor) pair) that the
ABFE calculations were actually run on.

These are **byte-identical to the corresponding files in
`../../02-building-poses-in-ApoDock-receptors/output/resulting_mols/`**, which is where
`run_abfe.py` sets its calculations up from. `abfe_analysis.ipynb` reads this copy rather
than step 02, so that the analysis keeps reproducing the published numbers if the step 02
output is ever regenerated: FEgrow pose generation is stochastic, so re-running it produces
different coordinates for the same molecules, which would otherwise silently invalidate the
RMSDs in `../output/abfe_df_processed.csv`.

Note that the step 06 and step 07 outputs currently committed (`lowest_rmsds.csv`,
`mol_scores_sorted.csv`, and the analyses downstream of them) were generated from an earlier
regeneration of the step 02 poses rather than from these, so they are not yet consistent
with the poses now in step 02. This is to be fixed by regenerating them.
