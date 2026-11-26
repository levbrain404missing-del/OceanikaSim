import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from importlib.metadata import distribution


class SDKProfileTests(unittest.TestCase):
    def test_installed_profile_matches_the_sdk_configuration_file(self):
        package = distribution("pioneer_sdk2")
        config = json.loads(package.locate_file("pioneer_sdk2/board_config.json").read_text(encoding="utf-8"))
        script = Path(__file__).with_name("sdk_profile.py")
        run = subprocess.run([sys.executable, "-B", str(script)], capture_output=True, check=True, timeout=5)
        report = json.loads(run.stdout)
        self.assertEqual(report["sdk_version"], package.version)
        self.assertEqual(report["board_profile"], config["board_name"])
        self.assertEqual(report["camera_driver"], config["cameras"]["driver"])
        self.assertIn("main", report["camera_types"])
        self.assertEqual(report["servo_configured"], bool(config["servo"]))

    def test_alternate_profile_and_missing_camera(self):
        from sdk_profile import read_profile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board_config.json"
            for cameras, driver, types in [
                ({"driver":"gstreamer","recorder":True,"main":{"name":"one"},"opt":{"name":"two"}},
                 "gstreamer", ["main","opt"]),
                (None, None, []),
            ]:
                path.write_text(json.dumps({"board_name":"fixture","cameras":cameras,"servo":{"channel":1}}))
                profile = read_profile(path,"fixture-version")
                self.assertEqual(profile["camera_types"],types)
                self.assertEqual(profile["camera_driver"],driver)
                self.assertTrue(profile["servo_configured"])
