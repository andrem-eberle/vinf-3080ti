from __future__ import annotations

import unittest

import vinf
from vinf.native import argmax_float32, native_version


class SmokeTests(unittest.TestCase):
    def test_import_and_config(self) -> None:
        config = vinf.EngineConfig()
        self.assertEqual(config.target_gpu, "rtx_3080_ti")
        self.assertFalse(config.speculative.enabled)

    def test_engine_shell(self) -> None:
        engine = vinf.InferenceEngine()
        with self.assertRaises(NotImplementedError):
            engine.generate("hello")

    def test_native_extension(self) -> None:
        self.assertEqual(native_version(), "vinf-native-0.0.1")
        self.assertEqual(argmax_float32([0.25, 3.5, -1.0, 2.0]), 1)


if __name__ == "__main__":
    unittest.main()
