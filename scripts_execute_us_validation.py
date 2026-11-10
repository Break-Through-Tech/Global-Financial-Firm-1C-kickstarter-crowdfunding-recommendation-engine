"""Execute only the training-only validation notebook; save outputs after success."""
import os
import sys
from pathlib import Path

# Set before importing numerical libraries in the new kernel.
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


class LocalKernel(KernelManager):
    def format_kernel_cmd(self, extra_arguments=None):
        return [sys.executable, '-m', 'ipykernel_launcher', '-f', self.connection_file,
                *(extra_arguments or [])]


def main():
    root = Path(__file__).resolve().parent
    name = sys.argv[1] if len(sys.argv) > 1 else '06_us_validation.ipynb'
    if name not in {'06_us_accuracy_experiments.ipynb', '06_us_validation.ipynb'}:
        raise ValueError('Only reviewed training-only notebooks can be executed')
    path = root / 'notebooks' / name
    original = path.read_bytes()
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=1800, kernel_name='python3',
                   kernel_manager_class=LocalKernel,
                   resources={'metadata': {'path': str(root)}}).execute()
    nbformat.validate(notebook)
    figures = [output for cell in notebook.cells if cell.cell_type == 'code'
               for output in cell.outputs if 'image/png' in output.get('data', {})]
    minimum = 1 if name == '06_us_accuracy_experiments.ipynb' else 3
    if len(figures) < minimum:
        raise RuntimeError('Required figures were not embedded; outputs not saved')
    if path.read_bytes() != original:
        raise RuntimeError('Notebook changed during execution; outputs not saved')
    temporary = path.with_suffix('.executed.tmp')
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    print('Executed and saved:', path)


if __name__ == '__main__':
    main()
