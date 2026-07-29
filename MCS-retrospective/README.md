## MCS Multitarget

The difference between [FEgrow-MCS](https://github.com/jthorton/polaris_fegrow_mcs) prospective submission and this retrospective submission is in this approach the receptor of the best MCS match is used instead of a single reference receptor.

## Usage

Install [FEgrow](https://cole-group.github.io/FEgrow/stable/installation/) or 

activate the environment from the fegrow-env.yaml by running:  

```conda env create -f fegrow-env.yaml ```

```conda activate fegrow-env```

For running the jupyter notebook:

```python -m ipykernel install --user --name=fegrow-env```

Step 1: Run sars_run_mcs/run_pose_gen-modified.py and full_run-MERS/run_pose_gen-modified.py to use FEgrow to grow the test molecules for sars and mers respectively in the sars pocket.

Step 2: Use build_best_sars.ipynb and build_best_mers.ipynb to align the poses to the respective receptor, generating best_sars.sdf and best_sars.sdf files.

Step 3: Run submit.ipynb to join the aligned best_sars.sdf, best_mers.sdf and submitted_default.sdf into a single final_submission.sdf.  

Step 4: Finally run RMSD_calculation.ipynb to generate the final % rmsd <2 A score.
