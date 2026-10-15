import json,pathlib,sys,contextlib,io
ROOT=pathlib.Path.cwd();sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
for f in ['notebooks/04_feature_engineering.ipynb','notebooks/us_only/04_feature_engineering.ipynb','notebooks/us_only/04b_us_eda.ipynb','notebooks/international_only/04_feature_engineering.ipynb']:
 e={'display':lambda *a,**k:None}
 with contextlib.redirect_stdout(io.StringIO()):
  for i,c in enumerate(json.loads((ROOT/f).read_text())['cells']):
   if c['cell_type']=='code':exec(''.join(c['source']),e)
 print('PASS fresh namespace',f,flush=True);plt.close('all')
 if 'state_rates' in e:print('US state',e['state_rates'].head(3).round(4).to_dict())
 if 'relative_bins' in e:
  from utils.features import rate_table
  print('US goal relative',rate_table(e['training'],e['relative_bins'],min_n=30).round(4).to_dict())
# Refit base preprocessors from current notebook code using training only.
P=ROOT/'data/Processed-Data'
for cohort in ['', 'us_only','international_only']:
 folder=P/cohort;c=pd.read_csv(folder/'DSI_kickstarterscrape_dataset_cleaned.csv');ids=pd.read_csv(folder/'train_project_ids.csv');x=c.set_index('project id').loc[ids['project id']].reset_index()
 if cohort=='international_only':
  from utils.international_scope import classify_locations
  x['country_region']=classify_locations(x.location).country_region.to_numpy()
 from sklearn.compose import ColumnTransformer
 from sklearn.preprocessing import StandardScaler,OneHotEncoder
 cat=['category','subcategory']+(['country_region'] if cohort=='international_only' else [])
 num=['goal','levels','duration']
 # Match stored ordering; no missing values exist in these columns.
 blocks=[('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat),('num',StandardScaler(),num)]
 if cohort:blocks=blocks[::-1]
 pp=ColumnTransformer(blocks)
 a=pp.fit_transform(x);saved=pd.read_csv(folder/'X_train_base_preprocessed.csv')
 assert a.shape==saved.shape and np.allclose(a,saved.to_numpy())
 print('PASS fresh fitted base matrix',cohort or 'combined',a.shape,flush=True)
