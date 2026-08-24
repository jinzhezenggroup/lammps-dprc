#!/usr/bin/env python3
"""CPU-only contract tests for the diagnostic DPA4c generator."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_unqualified_dpa4c",
    ROOT / "tools/generate_unqualified_dpa4c.py",
)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class UnqualifiedDPA4cGeneratorTest(unittest.TestCase):
    def test_configuration_is_compact_fp32_and_deterministic(self) -> None:
        config = GENERATOR.model_config(17)
        self.assertEqual(config["type_map"], ["C", "H", "HW", "O", "OW", "P"])
        self.assertEqual(config["atom_exclude_types"], [2, 4])
        self.assertEqual(
            [config["type_map"][index] for index in config["atom_exclude_types"]],
            ["HW", "OW"],
        )
        descriptor = config["descriptor"]
        self.assertEqual(
            {
                "type": descriptor["type"],
                "rcut": descriptor["rcut"],
                "channels": descriptor["channels"],
                "lmax": descriptor["lmax"],
                "n_radial": descriptor["n_radial"],
                "precision": descriptor["precision"],
                "seed": descriptor["seed"],
            },
            {
                "type": "dpa4c",
                "rcut": 6.0,
                "channels": 8,
                "lmax": 2,
                "n_radial": 8,
                "precision": "float32",
                "seed": 17,
            },
        )
        fitting = config["fitting_net"]
        self.assertEqual(fitting["neuron"], [32, 32])
        self.assertEqual(fitting["activation_function"], "silu")
        self.assertEqual(fitting["precision"], "float32")
        self.assertFalse(fitting["resnet_dt"])
        self.assertEqual(fitting["seed"], 18)


if __name__ == "__main__":
    unittest.main()
