"""Run the six reviewed active notebooks; commit notebook outputs only on success.

From repository root: .venv/bin/python docs/audit_evidence/execute_active_notebooks.py
The international training-CV CSV is the sole notebook data write; back it up.
"""
import os, sys, json, shutil, hashlib, time, tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path.cwd()
LOG=Path(tempfile.mkdtemp(prefix='team1c-notebooks-'))
print('Execution evidence and backups:', LOG, flush=True)
# Explicit executable avoids an unrelated system Python selected by a kernelspec.
class ReviewedKernel(KernelManager):
    def format_kernel_cmd(self, extra_arguments=None):
        return [sys.executable, '-m', 'ipykernel_launcher', '-f', self.connection_file, *(extra_arguments or [])]
paths=['notebooks/us_only/04_feature_engineering.ipynb',
       'notebooks/us_only/04b_us_eda.ipynb',
       'notebooks/us_only/05_modeling.ipynb',
       'notebooks/us_only/05a_safe_feature_check.ipynb',
       'notebooks/international_only/04_feature_engineering.ipynb',
       'notebooks/international_only/05_modeling.ipynb']
# Only international 05 writes: a local training-CV summary. Back it up.
summary=ROOT/'data/Processed-Data/international_only/training_cv_baselines.csv'
if summary.exists(): shutil.copy2(summary, LOG/'international-training_cv_baselines-before.csv')
records=[]
for name in paths:
    p=ROOT/name; original=p.read_bytes(); nb=nbformat.read(p,as_version=4)
    source=[(c.cell_type,c.source) for c in nb.cells]
    print('START',name,flush=True);start=time.time()
    try:
        client=NotebookClient(nb,timeout=1800,kernel_name='python3',
                kernel_manager_class=ReviewedKernel,resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        assert source==[(c.cell_type,c.source) for c in nb.cells]
        assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
        nbformat.validate(nb)
        temp=p.with_suffix('.executed.tmp');nbformat.write(nb,temp)
        assert p.read_bytes()==original, 'Notebook changed during execution'
        temp.replace(p)
        records.append({'notebook':name,'status':'saved','seconds':time.time()-start,
                        'code_cells':sum(c.cell_type=='code' for c in nb.cells)})
        print('SAVED',name,round(time.time()-start,2),flush=True)
    except Exception as exc:
        records.append({'notebook':name,'status':'failed; original preserved','error':repr(exc)})
        print('FAILED',name,repr(exc),flush=True)
    (LOG/'notebook-execution.json').write_text(json.dumps(records,indent=2))
