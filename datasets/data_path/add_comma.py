# Adding "," to the gt files
input_file_path = 'path/to/MOTS20-11/gt/gt_mask.txt'
output_file_path = 'path/to/gt_folder/00011.txt'

with open(input_file_path, 'r') as file, open(output_file_path, 'w') as output:
    for line in file:
        # Strip newline and any trailing spaces
        line = line.strip()
        # Split line by spaces
        elements = line.split()
        # Join elements with commas
        comma_separated_line = ','.join(elements)
        # Write the modified line to output file
        output.write(comma_separated_line + '\n')
