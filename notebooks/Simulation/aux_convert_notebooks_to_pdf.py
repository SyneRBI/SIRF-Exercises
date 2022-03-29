import shutil
from pathlib import Path 
import subprocess 

dpi_pdf = 300
notebook_path = Path("/home/sirfuser/devel/SIRF-Exercises/notebooks/Simulation/")
output_path = Path("/media/sf_CCPPETMR/")

list_notebooks = sorted(notebook_path.glob("*.ipynb"))

for f in list_notebooks:
    command = "jupyter-nbconvert --to html {}".format(f)
    subprocess.call(command, shell=True) 


list_htmls = sorted(notebook_path.glob("*.html"))

for f in list_htmls:
    fout = f.with_suffix(".pdf")
    command = "wkhtmltopdf --dpi {} {} {}".format(dpi_pdf, f, fout)
    subprocess.call(command, shell=True) 


list_pdfs = sorted(notebook_path.glob("*.pdf"))
for f in list_pdfs:
    shutil.copy(f, str(output_path))