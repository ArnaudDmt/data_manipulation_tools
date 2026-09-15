"""Run: .venv/bin/python scripts/test_paper_experiments.py"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from paper_results_scripts.plotContactPoses import load_rest_pose_data


class ContactPlotDataTest(unittest.TestCase):
    def test_current_names_and_overlap_stay_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output_data'
            output.mkdir()
            pd.DataFrame({'t': [0, .005, .01],
                          'debug_contactState_isSet_RightFootCenter': ['notSet', 'Set', 'Set']}
                         ).to_csv(output / 'logReplay.csv', sep=';', index=False)
            poses = pd.DataFrame({'t': [0, .005, .01], 'Mocap_position_x': [1, 2, 3],
                                  'Mocap_datasOverlapping': ['outside', 'Datas overlap', 'Datas overlap']})
            poses.to_csv(output / 'finalDataCSV.csv', sep=';', index=False)
            contacts, aligned = load_rest_pose_data(directory)
            self.assertEqual(list(contacts.t), list(aligned.t))
            self.assertEqual(list(aligned.Mocap_pos_x), [2, 3])
            self.assertIn('debug_contactState_isSet_RightFootForceSensor', contacts)
            poses.loc[1, 't'] = .007
            poses.to_csv(output / 'finalDataCSV.csv', sep=';', index=False)
            with self.assertRaises(ValueError):
                load_rest_pose_data(directory)


if __name__ == '__main__':
    unittest.main()
