"""Torch-guarded tests for the forward-dynamics control (STEP 4).

Verifies:
  * target-features gradient guard triggers when caller forgets .detach();
  * feature-shape mismatch raises;
  * constant-predictor baseline equals variance of the target features;
  * diagnostics classify a good state as passed and a collapsed state as failed;
  * append_diagnostics_jsonl produces a schema-consistent line.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:  # only run if torch is present (dreamer env on the cluster)
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this host")
class TestForwardDynamicsLoss(unittest.TestCase):

    def _shapes(self):
        return dict(B=4, T=8, D=16, A=7)

    def test_rejects_grad_target(self) -> None:
        from rebuttal.forward_dynamics import forward_dynamics_loss
        s = self._shapes()
        online = torch.randn(s["B"], s["T"], s["D"], requires_grad=True)
        target = torch.randn(s["B"], s["T"], s["D"], requires_grad=True)  # BAD
        actions = torch.randn(s["B"], s["T"], s["A"])
        pred = torch.nn.Linear(s["D"] + s["A"], s["D"])

        def predictor(x, a):
            return pred(torch.cat([x, a], dim=-1))
        with self.assertRaisesRegex(RuntimeError, "must be detached"):
            forward_dynamics_loss(online_features=online,
                                  target_features=target,
                                  predictor=predictor, actions=actions)

    def test_encoder_receives_grad_via_prediction_path(self) -> None:
        from rebuttal.forward_dynamics import forward_dynamics_loss
        s = self._shapes()
        online = torch.randn(s["B"], s["T"], s["D"], requires_grad=True)
        target = torch.randn(s["B"], s["T"], s["D"]).detach()
        actions = torch.randn(s["B"], s["T"], s["A"])
        pred = torch.nn.Linear(s["D"] + s["A"], s["D"])

        def predictor(x, a):
            return pred(torch.cat([x, a], dim=-1))
        loss, _ = forward_dynamics_loss(online_features=online,
                                        target_features=target,
                                        predictor=predictor, actions=actions)
        loss.backward()
        self.assertIsNotNone(online.grad)
        self.assertTrue(torch.any(online.grad != 0))

    def test_shape_mismatch_raises(self) -> None:
        from rebuttal.forward_dynamics import forward_dynamics_loss
        s = self._shapes()
        online = torch.randn(s["B"], s["T"], s["D"], requires_grad=True)
        target = torch.randn(s["B"], s["T"] + 1, s["D"]).detach()
        actions = torch.randn(s["B"], s["T"], s["A"])
        pred = torch.nn.Linear(s["D"] + s["A"], s["D"])

        def predictor(x, a):
            return pred(torch.cat([x, a], dim=-1))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            forward_dynamics_loss(online_features=online,
                                  target_features=target,
                                  predictor=predictor, actions=actions)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this host")
class TestForwardDynDiagnostics(unittest.TestCase):

    def test_passes_on_healthy_state(self) -> None:
        from rebuttal.forward_dynamics import (
            ForwardDynConfig, diagnose_forward_dynamics)
        cfg = ForwardDynConfig(target_mode="frozen_initial")
        online = torch.randn(4, 8, 32, requires_grad=True)
        # simulate encoder + predictor with grad
        encoder = torch.nn.Linear(32, 32)
        predictor = torch.nn.Linear(32, 32)
        online.sum().backward()  # sanity: online has grad
        # give encoder/predictor a nonzero grad
        for m in (encoder, predictor):
            for p in m.parameters():
                p.grad = torch.randn_like(p) * 1e-2
        pred_out = predictor(online[:, :-1])
        diag = diagnose_forward_dynamics(
            step=100, online_features=online, predictor_output=pred_out,
            constant_baseline_loss=1.0, loss=0.5, encoder=encoder,
            predictor=predictor, config=cfg,
        )
        self.assertTrue(diag.passed, msg=f"failed checks: {diag.failed_checks}")

    def test_flags_collapsed_features(self) -> None:
        from rebuttal.forward_dynamics import (
            ForwardDynConfig, diagnose_forward_dynamics)
        cfg = ForwardDynConfig(target_mode="frozen_initial")
        online = torch.zeros(4, 8, 32, requires_grad=True)   # zero variance
        encoder = torch.nn.Linear(32, 32)
        predictor = torch.nn.Linear(32, 32)
        pred_out = torch.zeros_like(online[:, :-1])
        diag = diagnose_forward_dynamics(
            step=100, online_features=online, predictor_output=pred_out,
            constant_baseline_loss=1.0, loss=1.5, encoder=encoder,
            predictor=predictor, config=cfg,
        )
        self.assertFalse(diag.passed)
        joined = " ".join(diag.failed_checks)
        self.assertIn("per_dim_variance_min", joined)
        self.assertIn("predictor_output_variance", joined)
        self.assertIn("encoder_grad_norm==0", joined)
        # loss !< baseline
        self.assertIn("!<", joined)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this host")
class TestBaseline(unittest.TestCase):

    def test_constant_predictor_baseline_equals_variance_of_target(self) -> None:
        from rebuttal.forward_dynamics import constant_predictor_baseline
        tgt = torch.randn(6, 8, 16)
        val = constant_predictor_baseline(tgt)
        # equals mean-squared deviation from the batch mean, i.e. var
        ref = float(((tgt[:, 1:] - tgt[:, 1:].mean(dim=(0, 1))) ** 2).mean())
        self.assertAlmostEqual(val, ref, places=5)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this host")
class TestDiagnosticsJSONL(unittest.TestCase):

    def test_writes_readable_jsonl(self) -> None:
        from rebuttal.forward_dynamics import (
            ForwardDynConfig, ForwardDynDiagnostics,
            append_diagnostics_jsonl)
        cfg = ForwardDynConfig()
        d = ForwardDynDiagnostics(
            step=1, per_dim_variance_mean=1.0, per_dim_variance_min=0.5,
            mean_pairwise_cosine=0.1, predictor_output_variance=0.9,
            encoder_grad_norm=0.4, predictor_grad_norm=0.7,
            loss=0.3, constant_baseline_loss=1.0,
        )
        d.evaluate(cfg)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "diag.jsonl"
            append_diagnostics_jsonl(d, p)
            rec = json.loads(p.read_text().strip())
            self.assertEqual(rec["step"], 1)
            self.assertTrue(rec["passed"])
            self.assertEqual(rec["failed_checks"], [])


if __name__ == "__main__":
    unittest.main()
