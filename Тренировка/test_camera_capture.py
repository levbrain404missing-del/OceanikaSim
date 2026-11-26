import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np


class CameraCaptureTests(unittest.TestCase):
    def test_recording_round_trip_and_release(self):
        from camera_capture import capture
        samples = [np.full((24,32,3),value,np.uint8) for value in (23, 230)]
        camera = Mock()
        camera.get_cv_frame.side_effect = samples
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "с камеры"
            capture(out,2,1,lambda:camera)
            files = sorted(out.glob("*.png"))
            self.assertEqual(len(files),2)
            for sample,file in zip(samples,files):
                decoded = cv2.imdecode(np.frombuffer(file.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(decoded,sample)
            with self.assertRaises(FileExistsError):
                capture(out,2,1,lambda:camera)
        camera.stop.assert_called_once()

    def test_timeout_preserves_first_frame_and_closes(self):
        from camera_capture import capture
        camera = Mock()
        camera.get_cv_frame.side_effect = [np.zeros((24,32,3),np.uint8),TimeoutError()]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "frames"
            with self.assertRaises(TimeoutError):
                capture(out,2,.2,lambda:camera)
            self.assertEqual(len(list(out.glob("*.png"))),1)
        camera.stop.assert_called_once()
