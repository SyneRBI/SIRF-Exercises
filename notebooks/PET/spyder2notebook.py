# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import sys
import os

def to_camel_case(snake_str):
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return ''.join(x.title() for x in components)

f = open(sys.argv[1], "r")
lines = f.readlines()
f.close()

cells = [] 
cell = []
    
for i in range(len(lines)):
    if lines[i][:3] == "#%%":
        cells.append(cell)
        cell = []
    cell.append(lines[i])
    

# create a notebook
notebook_cells = []
notebook_cell = {"cell_type": "code",
     "collapsed": False,
     "input": [
     ],
     "language": "python",
     "metadata": {},
     "outputs": []}
     
cell = []
for i in range(len(lines)):
    if lines[i][:3] == "#%%":
        notebook_cell['input'] = cell[:]
        notebook_cells.append(notebook_cell.copy())
        notebook_cell['input'] = []
        cell = []
    cell.append(lines[i])
# the last bit
notebook_cell['input'] = cell[:]
notebook_cells.append(notebook_cell.copy())
notebook_cell['input'] = []
        
    
#save to file
f = open("display_and_projection.t", "w")
a = json.dump(notebook_cells, f, indent=2)
f.close()   

out = to_camel_case(os.path.basename(sys.argv[1]).strip('.py')) + '.ipynb'

f = open(out, "r")
a = json.load(f)
a['worksheets'][0]['cells'] = notebook_cells
f.close()

print (os.path.basename(sys.argv[1])+" -> " + out)
fw = open(out, "w")
json.dump(a, fw, indent=2)
fw.close()
