mkdir sars_files

i=0
for f in Test_*.pdb; do
    cp "$f" "sars_files/sars_${i}.pdb"
    ((i++))
done
