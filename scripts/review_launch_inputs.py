"""Read-only membership and training-source audit; never reads reserved labels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
from utils.features import parse_tiers
from utils.us_modeling import load_us_training_data
from utils.us_scope import us_state_proxy_mask


def review():
    folder = ROOT / 'data/Processed-Data'
    membership = {}
    sets = {}
    for name, subdir, expected in [('global', '', (29288, 12553)),
                                   ('us', 'us_only', (26945, 11475)),
                                   ('international', 'international_only', (1427, 637))]:
        p = folder / subdir
        train = pd.read_csv(p / 'train_project_ids.csv', usecols=['project id'])['project id']
        test = pd.read_csv(p / 'test_project_ids.csv', usecols=['project id'])['project id']
        assert train.is_unique and test.is_unique
        assert set(train).isdisjoint(test)
        assert (len(train), len(test)) == expected
        clean_ids = pd.read_csv(p / 'DSI_kickstarterscrape_dataset_cleaned.csv', usecols=['project id'])['project id']
        assert clean_ids.is_unique and (set(train) | set(test)).issubset(clean_ids)
        sets[name] = (train, test)
        membership[name] = dict(train=len(train), reserved_ids_only=len(test),
                               ordered_train_ids_sha256=hashlib.sha256('\n'.join(train.astype(str)).encode()).hexdigest())
    for name in ['us', 'international']:
        for side in [0, 1]:
            parent = sets['global'][side]
            child = sets[name][side]
            assert parent[parent.isin(child)].tolist() == child.tolist()
    X, y, ids = load_us_training_data(ROOT)
    assert us_state_proxy_mask(X).all()
    # Inspect raw source fields only after restricting to official training IDs.
    raw = pd.read_csv(ROOT / 'data/Raw-data/DSI_kickstarterscrape_dataset.csv',
                      usecols=['project id', 'name', 'reward levels', 'levels'], encoding_errors='replace')
    raw = raw[raw['project id'].isin(ids)].drop_duplicates('project id').set_index('project id').loc[ids]
    tiers = X['reward levels'].map(parse_tiers)
    regex_tiers = X['reward levels'].map(lambda s: [float(v.replace(',', '')) for v in re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)', str(s))])
    source = dict(training_rows=len(X), success_rows=int(y.sum()), failure_rows=int((1-y).sum()),
                  reward_missing=int(X['reward levels'].isna().sum()),
                  empty_parsed_rewards=int(tiers.map(len).eq(0).sum()),
                  count_mismatch=int(tiers.map(len).ne(X['levels']).sum()),
                  parser_disagreement=int(sum(a != b for a, b in zip(tiers, regex_tiers))),
                  raw_reward_string_changes=int((raw['reward levels'].fillna('').to_numpy() != X['reward levels'].fillna('').to_numpy()).sum()),
                  title_missing=int(X['name'].isna().sum()),
                  title_replacement_characters=int(X['name'].str.contains('\ufffd', na=False).sum()),
                  fractional_duration=int(X['duration'].mod(1).ne(0).sum()),
                  categories=sorted(X['category'].unique()), subcategories=sorted(X['subcategory'].unique()),
                  tier_counts=tiers.map(len).describe().to_dict(),
                  duration=X['duration'].describe().to_dict(),
                  launch_snapshot_verified=False, global_currency_verified=False)
    env = dict(python=platform.python_version(), sklearn=sklearn.__version__,
               numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__,
               xgboost_installed=importlib.util.find_spec('xgboost') is not None,
               lightgbm_installed=importlib.util.find_spec('lightgbm') is not None)
    return dict(membership=membership, training_source=source, environment=env)


if __name__ == '__main__':
    print(json.dumps(review(), indent=2))
