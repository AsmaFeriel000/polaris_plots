def fix_sdf_headers(infile, outfile):
    """Insert proper 3-line headers before each molecule block and strip any extra blank lines."""
    with open(infile) as fin, open(outfile, "w") as fout:
        mol_index = 0
        blank_buffer = []

        for line in fin:
            stripped = line.strip()

            # Detect start of a molecule block (the counts line)
            if stripped.endswith(("V2000", "V3000")) and stripped and stripped[0].isdigit():
                # drop any blank lines seen right before the counts line
                blank_buffer.clear()

                # write a unique header for every molecule
                mol_index += 1
                fout.write(f"Mol_{mol_index}\n")   # molecule name
                fout.write("  RDKit  3D\n")        # program line
                fout.write("\n")                   # comment line (blank)
                fout.write(line)                   # counts line itself
            else:
                if stripped == "":
                    # buffer blank lines so they can be dropped if they precede a counts line
                    blank_buffer.append(line)
                else:
                    # flush buffered blanks + current nonblank line
                    fout.writelines(blank_buffer)
                    blank_buffer.clear()
                    fout.write(line)
                
fix_sdf_headers("output_every_9th.sdf", "mers-gnina-fixed.sdf")
