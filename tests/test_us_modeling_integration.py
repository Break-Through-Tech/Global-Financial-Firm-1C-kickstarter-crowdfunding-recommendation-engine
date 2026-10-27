"""Real-data checks skip only when the required local artifacts are absent."""
import os
from pathlib import Path
import unittest

from utils.us_modeling import EXPECTED_TRAIN_ROWS, load_us_training_data

ROOT = Path(os.environ.get('US_MODELING_INTEGRATION_ROOT', Path(__file__).resolve().parents[1]))
REQUIRED = [ROOT / 'data/Processed-Data/us_only' / filename for filename in
            ['DSI_kickstarterscrape_dataset_cleaned.csv', 'train_project_ids.csv']]


class USModelingIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(all(path.is_file() for path in REQUIRED),
                         'official processed U.S. data unavailable; integration test skipped')
    def test_official_training_membership_and_binary_labels(self):
        X, y, project_ids = load_us_training_data(ROOT)
        self.assertEqual(len(X), EXPECTED_TRAIN_ROWS)
        self.assertEqual(project_ids.nunique(), EXPECTED_TRAIN_ROWS)
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertEqual(len(X), len(y))
        self.assertNotIn('status', X.columns)
        self.assertNotIn('target', X.columns)
