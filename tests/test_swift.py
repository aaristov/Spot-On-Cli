from fastspt import swift
import pandas as pd
import unittest
from unittest import TestCase
import numpy as np


class TestSwiftImport(TestCase):

    def setUp(self):
        self._len = 50
        self.n_seg = 3
        self.n_track = 5
        self.sigma = 30
        xy = np.random.standard_normal((self._len, 2))
        frame = np.arange(self._len).reshape((self._len, 1))
        seg_id = np.linspace(
            1, self.n_seg, self._len).astype(int).reshape((self._len, 1))
        track_id = np.linspace(
            1, self.n_track, self._len).astype(int).reshape((self._len, 1))

        sigma = np.random.standard_gamma(self.sigma, (self._len, 1))

        self.data = np.concatenate(
            [xy, frame, seg_id, track_id, sigma], axis=1)
        self.df = pd.DataFrame(
            data=self.data,
            columns=[
                'x [nm]', 'y [nm]', 'frame', 'seg.id', 'track.id', 'sigma']
            )

    def test_data_integrity(self):
        [self.assertEqual(d, v)
            for d, v in zip(self.data.shape, (self._len, 6))]

    def test_import(self):
        tracks = swift.group_tracks_swift(self.df, min_len=0, max_len=np.inf)
        self.assertEqual(len(tracks), self.n_seg)

    def test_import_with_sigma(self):
        tracks = swift.group_tracks_swift(
            self.df,
            additional_columns=['sigma'],
            additional_scale=[1/1000.],
            additional_units=['um'],
            group_by='track.id',
            min_len=0
            )
        self.assertEqual(len(tracks), self.n_track)
        [self.assertAlmostEqual(np.mean(t.sigma), self.sigma/1000., 1)
            for t in tracks]


if __name__ == '__main__':
    unittest.main()
