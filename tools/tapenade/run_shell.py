import os

os.system('tapenade -reverse -head "square(x,y)" -output double functions.c')

# Assuming the generated files are `functions_b.c` and `run.c` contains the main function
os.system('gcc -I/usr/tapenade/ADFirstAidKit/ run.c functions.c double_b.c -o derivative')

# Run the program
os.system('./derivative')
