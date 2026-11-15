import nbformat
from nbclient import NotebookClient
nb = nbformat.read('gauss_flux_ablation.ipynb', as_version=4)
NotebookClient(nb, timeout=7200, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()
nbformat.write(nb, 'gauss_flux_ablation.ipynb')
print("DONE")
