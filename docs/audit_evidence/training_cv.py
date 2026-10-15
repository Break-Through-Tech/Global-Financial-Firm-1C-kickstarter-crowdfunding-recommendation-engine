import json,pathlib,sys,time
ROOT=pathlib.Path.cwd();sys.path.insert(0,str(ROOT))
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.base import clone
from utils.preprocessing import make_pipeline
from utils.us_launch_features import USLaunchFeatureBuilder
from utils.international_features import make_international_pipeline
def cells(f,indices):
 n=json.loads((ROOT/f).read_text()); e={}
 for i in indices:exec(''.join(n['cells'][i]['source']),e)
 return e,n
def report(cohort,name,pipe,e):
 t=time.time();s=cross_validate(pipe,e['X_train'],e['y_train'],cv=e['CV'],scoring=e['SCORING'],n_jobs=1,error_score='raise')
 print(json.dumps({'cohort':cohort,'model':name,'metrics':{k:float(v.mean()) for k,v in s.items() if k.startswith('test_')},'auc_folds':s['test_roc_auc'].tolist(),'seconds':time.time()-t}),flush=True)
e,n=cells('notebooks/us_only/05_modeling.ipynb',[1])
exec(''.join(n['cells'][2]['source']).split('rows = []')[0],e)
for name,(model,strategy) in e['MODELS'].items():report('us',name,make_pipeline(model,categorical_strategy=strategy,**e['FEATURE_POLICY']),e)
for name in ['logistic','hist_gradient_boosting']:
 model,strategy=e['MODELS'][name]
 pipe=make_pipeline(clone(model),categorical_strategy=strategy,**e['FEATURE_POLICY'])
 pipe.set_params(features=USLaunchFeatureBuilder(**e['FEATURE_POLICY']))
 report('us_extra',name,pipe,e)
e,n=cells('notebooks/international_only/05_modeling.ipynb',[1])
exec(''.join(n['cells'][2]['source']).split('rows = []')[0],e)
for name,model in e['MODELS'].items():report('international',name,make_international_pipeline(model,categorical_strategy='onehot',**e['FEATURE_POLICY']),e)
e,n=cells('notebooks/05b_pre_tuning_validation.ipynb',[1,3,5])
exec(''.join(n['cells'][9]['source']).split('records = []')[0],e)
for scenario,kwargs in e['SCENARIOS'].items():
 for name,model in e['MODELS'].items():report('combined_05b',scenario+'/'+name,make_pipeline(model,**kwargs),e)
