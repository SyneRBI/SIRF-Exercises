# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import os
import nbformat

f = open(sys.argv[1], "r")
lines = f.readlines()
f.close()

cell = []

# create a list of notebook cells
notebook_cells = []
     
cell = []
for i in range(len(lines)):
    if lines[i][:3] == "#%%":
        notebook_cells.append(nbformat.v4.new_code_cell(cell[:]))
        cell = []
    cell.append(lines[i])
# the last bit
notebook_cells.append( nbformat.v4.new_code_cell(cell[:]))

nb =  nbformat.v4.new_notebook(cells=notebook_cells)
   
#save to file

print (os.path.basename(sys.argv[1])+" -> " + sys.argv[2])
fw = open(sys.argv[2], "w")

nbformat.v4.nbjson.write(fp=fw, nb = nb)
