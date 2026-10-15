import contextlib, io, json, pathlib, sys, hashlib, platform
import numpy as np
import pandas as pd
ROOT=pathlib.Path.cwd(); sys.path.insert(0,str(ROOT))
from utils.international_scope import classify_locations, classify_location
from utils.us_scope import us_state_proxy_mask
P=ROOT/'data/Processed-Data'
raw=pd.read_csv(ROOT/'data/Raw-data/DSI_kickstarterscrape_dataset.csv',encoding_errors='replace')
clean=pd.read_csv(P/'DSI_kickstarterscrape_dataset_cleaned.csv')
def out(k,v): print(k, v, flush=True)
out('versions',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__})
import sklearn;out('sklearn',sklearn.__version__)
out('raw',{'shape':raw.shape,'ids':raw['project id'].nunique(),'duplicate_ids':int(raw['project id'].duplicated().sum()),'exact_duplicates':int(raw.duplicated().sum()),'status':raw.status.value_counts().to_dict(),'missing':raw.isna().sum().to_dict(),'dtypes':raw.dtypes.astype(str).to_dict()})
# Replay only cleaning cells that calculate/validate: never Drive or writes.
n=json.loads((ROOT/'notebooks/02_clean_data.ipynb').read_text())
env={'df':raw.copy(),'display':lambda *a,**k:None}
exec('import pandas as pd\nimport numpy as np\nimport html,re',env)
with contextlib.redirect_stdout(io.StringIO()):
 for i in range(4,76):
  c=n['cells'][i]
  if c['cell_type']=='code':exec(''.join(c['source']),env)
rebuilt=env['clean_df']
pd.testing.assert_frame_equal(pd.read_csv(io.StringIO(rebuilt.to_csv(index=False))),clean)
out('cleaning_replay','PASS exact CSV-equivalent equality')
out('clean',{'shape':clean.shape,'status':clean.status.value_counts().to_dict(),'missing':clean.isna().sum().to_dict(),'numeric':clean[['goal','duration','levels']].describe().to_dict(),'replacement':{c:int(clean[c].astype(str).str.contains('\ufffd',regex=False).sum()) for c in clean.select_dtypes(include=['object','string','str'])}})
geo=classify_locations(clean.location)
out('geo',geo.groupby(['scope','evidence'],dropna=False).size().to_dict())
out('unknowns',clean.loc[geo.scope.eq('unknown'),'location'].value_counts(dropna=False).to_dict())
out('territories',clean.loc[geo.scope.eq('us_territory'),'location'].value_counts(dropna=False).to_dict())
out('countries',geo.country_region[geo.scope.eq('explicit_non_us')].value_counts().to_dict())
tr=pd.read_csv(P/'train_project_ids.csv'); te=pd.read_csv(P/'test_project_ids.csv',usecols=['project id'])
assert set(tr['project id']).isdisjoint(te['project id'])
settled=clean.status.isin(['successful','failed'])
assert set(tr['project id'])|set(te['project id'])==set(clean.loc[settled,'project id'])
from sklearn.model_selection import train_test_split
a,b=train_test_split(clean.loc[settled,'project id'],test_size=.3,random_state=42,stratify=clean.loc[settled,'status'].eq('successful'))
assert a.tolist()==tr['project id'].tolist() and b.tolist()==te['project id'].tolist()
out('official_split_replay','PASS exact ordered IDs; reserved target files not opened')
from utils.utils import get_data_split
historical_train,historical_test,_,_=get_data_split(raw)
out('historical_raw_split_overlapping_ids',len(set(historical_train['project id']) & set(historical_test['project id'])))
for label,folder in [('combined',P),('us',P/'us_only'),('international',P/'international_only')]:
 reserved=pd.read_csv(folder/'test_project_ids.csv',usecols=['project id'])
 out(label+'_reserved_ids_exposed_to_raw_training',len(set(reserved['project id']) & set(historical_train['project id'])))
assert not (geo.scope.eq('explicit_non_us') & us_state_proxy_mask(clean)).any()
for label,folder,scope in [('combined',P,None),('us',P/'us_only','us_candidate'),('international',P/'international_only','explicit_non_us')]:
 d=pd.read_csv(folder/'DSI_kickstarterscrape_dataset_cleaned.csv')
 expected=clean if scope is None else clean.loc[geo.scope.eq(scope)]
 pd.testing.assert_frame_equal(d.reset_index(drop=True),expected.reset_index(drop=True))
 ti=pd.read_csv(folder/'train_project_ids.csv');vi=pd.read_csv(folder/'test_project_ids.csv',usecols=['project id'])
 ids=set(d.loc[d.status.isin(['successful','failed']),'project id'])
 assert set(ti['project id'])|set(vi['project id'])==ids
 assert ti['project id'].tolist()==tr.loc[tr['project id'].isin(ids),'project id'].tolist()
 assert vi['project id'].tolist()==te.loc[te['project id'].isin(ids),'project id'].tolist()
 rows=d.set_index('project id').loc[ti['project id']]
 assert rows.status.eq('successful').astype(int).tolist()==ti.target.tolist()
 assert pd.read_csv(folder/'y_train.csv').target.astype(int).tolist()==ti.target.tolist()
 out(label,{'all':len(d),'statuses':d.status.value_counts().to_dict(),'settled':len(ids),'train':len(ti),'test':len(vi),'train_success':int(ti.target.sum()),'train_failed':int((1-ti.target).sum()),'train_rate':ti.target.mean(),'all_settled_rate':d.loc[d.status.isin(['successful','failed']),'status'].eq('successful').mean()})
 import joblib
 try: prep=joblib.load(folder/'base_preprocessor.joblib')
 except Exception as exc:
  out(label+'_artifact_load_failure',repr(exc));continue
 saved=pd.read_csv(folder/'X_train_base_preprocessed.csv')
 recomputed=prep.transform(rows.reset_index())
 assert np.allclose(saved.to_numpy(),recomputed)
 out(label+'_base',{'shape':saved.shape,'feature_names_match':list(saved.columns)==list(prep.get_feature_names_out()),'finite':bool(np.isfinite(saved.to_numpy()).all())})
 for name,transformer,cols in prep.transformers_:
  if hasattr(transformer,'named_steps'):
   for step in transformer.named_steps.values():
    if hasattr(step,'mean_'):
     assert np.allclose(step.mean_,rows[cols].mean())
     assert np.allclose(step.var_,rows[cols].var(ddof=0))
out('scope_status',pd.crosstab(geo.scope,clean.status).to_dict())
for s in ['St Thomas, Virgin Islands, U.S.','St Thomas, Virgin Islands of the United States','CA','City, CA','City, ON','Istanbul, Turkey','Gaza City, Palestinian Territories','Tbilisi, Georgia','Kinshasa, Congo','Charlottesville, Virginia']:
 out('probe',(s,classify_location(s)))
out('virgin_observed',clean.loc[clean.location.str.contains('Virgin',case=False,na=False),['project id','location']].to_dict('records'))
out('explicit_US_not_proxy',clean.loc[geo.scope.eq('explicit_us'),['project id','location']].to_dict('records'))
out('source_hashes',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/Raw-data/DSI_kickstarterscrape_dataset.csv',P/'DSI_kickstarterscrape_dataset_cleaned.csv',P/'train_project_ids.csv']})
