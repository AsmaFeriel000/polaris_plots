### To activate and use gnina-env environment, use:

```conda env create -f gnina-env.yaml```
```conda activate gnina-env```


1) To convert a list of smiles to 3d sdf file, run prep3D-mers.ipynb

2) Run following commands to use gnina for docking sdf file you just created


	### line below is to protonate reference lig

	obabel -isdf lig.sdf -osdf -O lig-H.sdf -p 7

	obabel -isdf lig-3D.sdf -osdf -O lig-3D-H.sdf -p 7

	### move gnina executable to current directory then run: 
	./gnina -r complex-mers.pdb -l lig-3D-H.sdf --autobox_ligand lig-H.sdf --seed 0 --no_gpu -o docked.sdf


3) Run extract_sdf.py
4) Then, run fix-gnina-output.py and the output file generated is used to create cdf plot. 
