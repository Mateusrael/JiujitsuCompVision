"""Opt-in CPU checks for pose architectures, masking, and saved model definitions."""

import json
import os
import unittest


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Set JIUJITSU_RUN_TORCH_TESTS=1 for CPU model checks")
class PoseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from src.Modules.models import architecture_config, build_model
        cls.torch = torch
        cls.architecture_config = staticmethod(architecture_config)
        cls.build_model = staticmethod(build_model)
        torch.set_num_threads(1)

    def setUp(self):
        self.torch.manual_seed(27)

    def attention_model(self, **overrides):
        parameters = dict(attention_dim=32, attention_heads=4, attention_layers=2,
                          attention_mlp_dim=64,
                          attention_pooling="mean")
        parameters.update(overrides)
        return self.build_model(self.architecture_config("pose-attention", **parameters))

    def observed_features(self, batch=3):
        features = self.torch.randn(batch, 34, 3)
        features[:, :, 2] = self.torch.rand(batch, 34) * 0.9 + 0.1
        return features.reshape(batch, 102)

    def test_reference_mlp_design_and_default_dropout(self):
        expected = {"type": "pose_mlp", "input_features": 102, "hidden": [102, 34],
                    "num_classes": 10, "pose_preprocessing": "pair_bbox_v1", "dropout": 0.0}
        self.assertEqual(self.architecture_config("pose"), expected)
        model = self.build_model(expected)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 14358)

    def test_wide_mlp_uses_requested_layers_and_relu_hidden_activations(self):
        nn = self.torch.nn
        model = self.build_model(self.architecture_config("pose-wide"))
        layers = list(model.network)
        self.assertEqual([type(layer) for layer in layers],
                         [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU,
                          nn.Linear, nn.ReLU, nn.Linear])
        self.assertEqual([(layer.in_features, layer.out_features) for layer in layers
                          if isinstance(layer, nn.Linear)],
                         [(102, 512), (512, 128), (128, 32), (32, 10)])
        logits = model(self.observed_features())
        self.assertEqual(tuple(logits.shape), (3, 10))
        loss = nn.functional.cross_entropy(logits, self.torch.tensor([0, 3, 9]))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None
                            and self.torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_attention_has_34_identity_embeddings_and_trainable_attention(self):
        model = self.attention_model()
        self.assertEqual(tuple(model.position_embedding.shape), (1, 34, 32))
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(model.input_projection.in_features, 3)
        self.assertEqual(model.input_projection.out_features, 32)
        logits = model(self.observed_features())
        self.assertEqual(tuple(logits.shape), (3, 10))
        self.torch.nn.functional.cross_entropy(logits, self.torch.tensor([0, 3, 9])).backward()
        parameters = [model.position_embedding, model.input_projection.weight,
                      model.classifier.weight]
        parameters.extend(block.attention.in_proj_weight for block in model.blocks)
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(self.torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_joint_identity_comes_from_position_embedding(self):
        torch = self.torch
        model = self.attention_model().eval()
        features = self.observed_features().reshape(3, 34, 3)
        permutation = torch.arange(33, -1, -1)
        with torch.inference_mode():
            original = model(features.reshape(3, 102))
            reassigned = model(features[:, permutation].reshape(3, 102))
            self.assertFalse(torch.allclose(original, reassigned, atol=1e-6, rtol=1e-5))
            # Move each identity embedding with its joint: order alone must not
            # change a mean-pooled set of the same joint identities and values.
            model.position_embedding.copy_(model.position_embedding[:, permutation].clone())
            reordered = model(features[:, permutation].reshape(3, 102))
        torch.testing.assert_close(original, reordered, atol=1e-6, rtol=1e-5)

    def test_attention_blocks_normalize_before_each_branch_and_preserve_residual(self):
        torch = self.torch
        block = self.attention_model().blocks[0]
        order = []
        handles = [module.register_forward_pre_hook(
            lambda module, inputs, name=name: order.append(name))
            for name, module in (("norm1", block.norm1), ("attention", block.attention),
                                 ("norm2", block.norm2), ("mlp", block.mlp))]
        tokens = torch.randn(2, 34, 32)
        with torch.no_grad():
            block.attention.out_proj.weight.zero_()
            block.attention.out_proj.bias.zero_()
            final_linear = [module for module in block.mlp
                            if isinstance(module, torch.nn.Linear)][-1]
            final_linear.weight.zero_()
            final_linear.bias.zero_()
            output = block(tokens, torch.zeros(2, 34, dtype=torch.bool))
        for handle in handles:
            handle.remove()
        self.assertEqual(order, ["norm1", "attention", "norm2", "mlp"])
        torch.testing.assert_close(output, tokens, rtol=0, atol=0)

    def test_missing_joint_coordinates_do_not_affect_predictions(self):
        torch = self.torch
        model = self.attention_model().eval()
        features = self.observed_features().reshape(3, 34, 3)
        features[:, 5:9, :] = 0
        features[1, 17:, :] = 0
        changed = features.clone()
        missing = features[:, :, 2] <= 0
        changed[:, :, :2][missing] = torch.randn_like(changed[:, :, :2][missing]) * 10000
        changed[:, :, 2][missing] = -1
        with torch.inference_mode():
            baseline = model(features.reshape(3, 102))
            masked = model(changed.reshape(3, 102))
        torch.testing.assert_close(baseline, masked, atol=1e-6, rtol=1e-5)

    def test_one_and_both_missing_athletes_remain_finite(self):
        torch = self.torch
        for pooling in ("mean", "cls"):
            with self.subTest(pooling=pooling):
                model = self.attention_model(attention_pooling=pooling)
                features = self.observed_features().reshape(3, 34, 3)
                features[0, :17] = 0
                features[1, 17:] = 0
                features[2] = 0
                logits = model(features.reshape(3, 102))
                self.assertTrue(torch.isfinite(logits).all())
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3, 9]))
                loss.backward()
                self.assertTrue(all(torch.isfinite(parameter.grad).all()
                                    for parameter in model.parameters()
                                    if parameter.grad is not None))

    def test_pose_model_configuration_survives_json_and_state_round_trip(self):
        torch = self.torch
        for name in ("pose", "pose-wide", "pose-attention"):
            with self.subTest(model=name):
                config = self.architecture_config(name, attention_dim=48, attention_heads=3,
                    attention_layers=1, attention_mlp_dim=80, dropout=0.1,
                    attention_dropout=0.2, attention_mlp_dropout=0.3,
                    attention_pooling="cls")
                if name == "pose-attention":
                    self.assertEqual(config["attention_dropout"], 0.2)
                    self.assertEqual(config["mlp_dropout"], 0.3)
                model = self.build_model(config).eval()
                restored = self.build_model(json.loads(json.dumps(config))).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                features = self.observed_features()
                with torch.inference_mode():
                    torch.testing.assert_close(model(features), restored(features), rtol=0, atol=0)

    def test_dropout_follows_each_hidden_mlp_activation(self):
        torch = self.torch
        for name in ("pose", "pose-wide"):
            with self.subTest(model=name):
                model = self.build_model(self.architecture_config(name, dropout=0.5)).train()
                activations, linear_inputs = [], []
                handles = []
                for layer in model.network:
                    if isinstance(layer, torch.nn.ReLU):
                        handles.append(layer.register_forward_hook(
                            lambda module, inputs, output: activations.append(output.detach().clone())))
                    elif isinstance(layer, torch.nn.Linear):
                        handles.append(layer.register_forward_pre_hook(
                            lambda module, inputs: linear_inputs.append(inputs[0].detach().clone())))
                model(self.observed_features(batch=16))
                for handle in handles:
                    handle.remove()
                self.assertEqual(len(activations), len(linear_inputs) - 1)
                for activation, next_input in zip(activations, linear_inputs[1:]):
                    self.assertTrue(torch.all((next_input == 0) | (next_input == activation * 2)))
                    self.assertTrue(torch.any((activation > 0) & (next_input == 0)))

    def test_dropout_is_stochastic_only_during_training_for_all_models(self):
        torch = self.torch
        for name in ("pose", "pose-wide", "pose-attention", "image"):
            with self.subTest(model=name):
                config = self.architecture_config(name, dropout=0.4, attention_dim=32,
                    attention_heads=2, attention_layers=1, attention_mlp_dim=64)
                if name == "pose-attention":
                    self.assertEqual(config["attention_dropout"], 0.4)
                    self.assertEqual(config["mlp_dropout"], 0.4)
                else:
                    self.assertEqual(config["dropout"], 0.4)
                model = self.build_model(config).train()
                features = torch.randn(4, 3, 32, 32) if name == "image" else self.observed_features(4)
                if name == "image":
                    self.assertFalse(model.backbone.training)
                    self.assertTrue(model.dropout.training)
                    self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
                    buffers = {key: value.clone() for key, value in model.backbone.named_buffers()}
                first, second = model(features), model(features)
                self.assertFalse(torch.equal(first, second))
                if name == "image":
                    for key, value in model.backbone.named_buffers():
                        torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
                model.eval()
                with torch.inference_mode():
                    torch.testing.assert_close(model(features), model(features), rtol=0, atol=0)

    def test_attention_and_mlp_dropout_configure_only_their_respective_branches(self):
        torch = self.torch
        config = self.architecture_config("pose-attention", attention_dropout=0.1,
                                          attention_mlp_dropout=0.3)
        self.assertNotIn("dropout", config)
        self.assertEqual(config["attention_dropout"], 0.1)
        self.assertEqual(config["mlp_dropout"], 0.3)
        model = self.build_model(config)
        for block in model.blocks:
            self.assertEqual(block.attention.dropout, 0.1)
            self.assertEqual(block.attention_dropout.p, 0.1)
            self.assertEqual([layer.p for layer in block.mlp
                              if isinstance(layer, torch.nn.Dropout)], [0.3, 0.3])

    def test_attention_and_mlp_dropout_can_each_be_disabled_independently(self):
        torch = self.torch
        for attention_rate, mlp_rate in ((0.4, 0.0), (0.0, 0.4), (0.0, 0.0)):
            with self.subTest(attention_dropout=attention_rate, mlp_dropout=mlp_rate):
                # Explicit zero must override even a nonzero shared fallback.
                model = self.attention_model(dropout=0.8, attention_dropout=attention_rate,
                                             attention_mlp_dropout=mlp_rate).train()
                block = model.blocks[0]
                tokens = torch.randn(3, 34, 32)
                attention_first = block.attention(tokens, tokens, tokens, need_weights=False)[0]
                attention_second = block.attention(tokens, tokens, tokens, need_weights=False)[0]
                self.assertEqual(torch.equal(attention_first, attention_second), attention_rate == 0)
                self.assertEqual(torch.equal(block.mlp(tokens), block.mlp(tokens)), mlp_rate == 0)
                features = self.observed_features()
                self.assertEqual(torch.equal(model(features), model(features)),
                                 attention_rate == 0 and mlp_rate == 0)
                model.eval()
                with torch.inference_mode():
                    torch.testing.assert_close(model(features), model(features), rtol=0, atol=0)

    def test_invalid_dropout_is_rejected_for_every_architecture(self):
        for name in ("pose", "pose-wide", "pose-attention", "image"):
            for dropout in (-0.1, 1, float("nan"), float("inf"), True):
                with self.subTest(model=name, dropout=dropout):
                    with self.assertRaises(ValueError):
                        self.architecture_config(name, dropout=dropout)

    def test_invalid_attention_architectures_are_rejected(self):
        cases = ({"attention_dim": 0}, {"attention_dim": 31, "attention_heads": 4},
                 {"attention_heads": 0}, {"attention_layers": 0},
                 {"attention_mlp_dim": -1}, {"attention_dropout": -0.1},
                 {"attention_dropout": 1.0}, {"attention_dropout": float("nan")},
                 {"attention_mlp_dropout": -0.1}, {"attention_mlp_dropout": 1.0},
                 {"attention_mlp_dropout": float("nan")},
                 {"attention_pooling": "last"})
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.build_model(self.architecture_config("pose-attention", **changes))
        config = self.architecture_config("pose-wide")
        config["hidden"] = [512, 128, 16]
        with self.assertRaises(ValueError):
            self.build_model(config)
        config = self.architecture_config("pose-attention")
        config["unrecognized_parameter"] = True
        with self.assertRaises(ValueError):
            self.build_model(config)


if __name__ == "__main__":
    unittest.main()
