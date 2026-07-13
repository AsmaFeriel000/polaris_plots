- Need to install typer as well as a3fe, and openeye toolkits with
```
mamba install -c conda-force -c openeye typer openeye-toolkits spyrmsd
```
Need to manually rename an annoying H in the NSER residues in the input pdb files:

Line 11 H -> H1



- Run a given mol with:
```