import subprocess
import sys
import unittest
from pathlib import Path

LIGHT_MODELS = ("shufflenet_v2_x0_5", "mobilenet_v3_small", "mnasnet0_5", "efficientnet_b0")
ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ImportError:
    torch = None


class LightArchitectureTests(unittest.TestCase):
    def test_cli_accepts_light_models(self):
        for architecture in LIGHT_MODELS:
            with self.subTest(architecture=architecture):
                proc = subprocess.run(
                    [sys.executable, "-m", "reproduction", "check",
                     "--architecture", architecture, "--data", "missing-test-data"],
                    cwd=ROOT, capture_output=True, text=True,
                )
                self.assertEqual(proc.returncode, 1, proc.stderr)
                self.assertIn("metadata.csv", proc.stderr)

    @unittest.skipIf(torch is None, "PyTorch necessário para o teste das redes")
    def test_light_models_support_training_and_last_layer_features(self):
        from reproduction.architectures import MODEL_SPECS, get_embedding, make_model

        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(2)
        batch = torch.zeros(2, 3, 224, 224)
        for architecture in LIGHT_MODELS:
            with self.subTest(architecture=architecture):
                model = make_model(architecture, 2, pretrained=False)
                head = model.get_submodule(".".join(MODEL_SPECS[architecture][2]))
                logits = model(batch)
                self.assertEqual(logits.shape, (2, 2))
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                self.assertTrue(torch.isfinite(head.weight.grad).all().item())
                model.eval()
                with torch.inference_mode():
                    embedding = get_embedding(model, batch, architecture)
                    self.assertEqual(embedding.shape, (2, head.in_features))
                    torch.testing.assert_close(head(embedding), model(batch))
