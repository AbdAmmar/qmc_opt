

import globals

def append_to_output(text):
    with open(globals.file_out, 'a') as file:
        file.write(text + '\n')
