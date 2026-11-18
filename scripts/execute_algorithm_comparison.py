"""Execute only the additional algorithm notebook with the invoking interpreter."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
import nbformat
from nbclient import NotebookClient
from scripts_execute_us_validation import LocalKernel

path = ROOT / 'notebooks/06_us_algorithm_comparison.ipynb'
source = path.read_bytes()
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=1800, kernel_name='python3',
               kernel_manager_class=LocalKernel,
               resources={'metadata': {'path': str(ROOT)}}).execute()
nbformat.validate(notebook)
assert path.read_bytes() == source, 'Notebook changed during execution'
temporary = path.with_suffix('.executed.tmp')
nbformat.write(notebook, temporary)
temporary.replace(path)
print('Executed and saved', path)
