"""The prepared reserved notebook must stop before any dataset/model loading."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch


class ReservedProtocolTests(unittest.TestCase):
    def test_empty_team_approval_blocks_both_cells(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'notebooks/08_us_reserved_evaluation.ipynb').read_text())
        namespace = {}
        with patch('pandas.read_csv', side_effect=AssertionError('Dataset read before approval')), patch(
                'joblib.load', side_effect=AssertionError('Model load before approval')):
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    with self.assertRaisesRegex(RuntimeError, 'approval'):
                        exec(''.join(cell['source']), namespace)
