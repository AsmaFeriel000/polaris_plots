mkdir mers_files

i=0
for f in Test_*.pdb; do
    cp "$f" "mers_files/mers_${i}.pdb"
    ((i++))
done