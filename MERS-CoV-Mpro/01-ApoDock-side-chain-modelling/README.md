## Generating receptors with modelled side chains:

[ApoDock](https://github.com/AsmaFeriel000/ApoDock_public) was installed and used to model files in ./inputs, generating the ensemble of receptors in ./output

The environment used is attached for convenience (apodock-env.yaml) and can be activated by the following command:

```
conda env create -f apodock-env.yaml
```
```
conda activate apodock-env
```

## Protonating the modelled receptors using Chimera:

1- generate a new environment and install [chimera](https://www.cgl.ucsf.edu/chimera/download.html)

2- If Chimera is installed in: /home/yourname/chimera/bin then you would run: 

```export PATH=$PATH:/home/yourname/chimera/bin``` 

3- run protonate-all.py with chimera command not python (using: ```chimera --nogui --script protonate-all.py```)
