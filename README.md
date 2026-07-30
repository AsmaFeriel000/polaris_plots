## The pipeline

The code that was used to produce the plots of the paper is all included here, and should be reproducible. The pipeline is performed in several steps, with instructions for how to run each step in the README file in each directory.

## Directory overview

| Directory | Contents |
| --- | --- |
| ```SARS-CoV-2-Mpro``` | The full pipeline for SARS-CoV-2 Mpro, in numbered steps (```01``` to ```07```) |
| ```MERS-CoV-Mpro``` | The same pipeline for MERS-CoV Mpro, plus the ABFE rescoring in ```08``` |
| ```MCS-retrospective``` | Retrospective pose generation using the maximum common substructure approach |
| ```released-receptor-structures``` | The released (unblinded) receptor structures and test molecules |
| ```plotting``` | ```plots.ipynb```, which reproduces the paper figures |

The numbered steps within each target directory are run in order:

1. ```01-ApoDock-side-chain-modelling``` - model the pocket side chains with ApoDock
2. ```02-building-poses-in-ApoDock-receptors``` - build the poses with FEgrow
3. ```03-Building-complexes``` - assemble the protein-ligand complexes
4. ```04-Figure_5```, ```05-Figure_6```, ```06-Figure_8```, ```07-Figure_9``` - the analyses behind each figure
5. ```08-ABFE-rescoring``` (MERS only) - rescore the FEgrow poses with absolute binding free energies using [a3fe](https://github.com/michellab/a3fe), and analyse the results. The raw a3fe calculation tree is hundreds of GB and is not committed; the processed results it produces are, so the analysis notebook reproduces without it. This step uses its own environment, described in its README.

## Python environment

``` new-env.yaml``` should be used as the main environment for running all the scripts unless a different environment is specified in the directory's README.

Create and activate the environment using the commands below:

```conda env create -f new-env.yaml```

```conda activate new-env```


