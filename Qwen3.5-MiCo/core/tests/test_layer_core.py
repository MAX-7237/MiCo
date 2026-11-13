"""CPU contracts for the public MiCo QK recipe."""

import copy
from fractions import Fraction
import importlib.util
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_test_utils import model, inputs, GEN, torch
from unified_core import install_pruning
from budget_schedule import mico_budget


class LayerCoreTests(unittest.TestCase):

    def test_all_layer_budgets_and_rejected_alpha(self):
        for target in (128, 256):
            for cut in range(1, 16):
                result = mico_budget(target, 784, cut_after_layer=cut)
                self.assertEqual(result["n1"], 2 * target)
                self.assertEqual(
                    result["n2"], round(Fraction(32 * target - cut * 2 * target, 32 - cut))
                )
        for cut in (0, 16, True, 4.0):
            with self.assertRaises(ValueError):
                mico_budget(128, 784, cut_after_layer=cut)
        for pair in ((3, 2), (5, 4)):
            with self.assertRaises(ValueError):
                mico_budget(128, 784, alpha_num=pair[0], alpha_den=pair[1])

    @torch.no_grad()
    def test_qk_twelve_and_s1_have_distinct_trace(self):
        m = model()
        data = inputs(12)
        ctl = install_pruning(m, "mico", 8, cut_after_layer=12)
        m.generate(**data, **GEN)
        self.assertEqual(ctl.last_record["score_source_layer"], 12)
        self.assertEqual(ctl.last_record["score_query_scope"], "all_nonvisual")
        ctl.remove()
        ctl = install_pruning(m, "mico_s1", 8)
        m.generate(**data, **GEN)
        self.assertIsNone(ctl.last_record["score_source_layer"])
        self.assertIsNone(ctl.last_record["drop_after_layer"])
        self.assertEqual(ctl.last_record["layer_visual_counts"], [8] * 32)
        ctl.remove()


if __name__ == "__main__":
    unittest.main()
