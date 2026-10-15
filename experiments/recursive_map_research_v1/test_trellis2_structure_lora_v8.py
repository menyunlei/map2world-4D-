"""CPU-only synthetic LoRA tests; no native weights, network, or GPU use."""

import copy
import io
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from trellis2_structure_lora_v8 import (
    LORA_ALPHA, LORA_BLOCKS, LORA_DROPOUT, LORA_RANK, PROJECTION_SHAPES,
    LoRALinear, attach_lora, load_lora_state_dict, lora_state_dict,
    trainable_parameters,
)


class FakeCrossAttention(nn.Module):
    def __init__(self, native_width):
        super().__init__()
        width, context_width = (1536, 1024) if native_width else (8, 6)
        self.to_q = nn.Linear(width, width, device="cpu")
        self.to_kv = nn.Linear(context_width, 2 * width, device="cpu")
        self.to_out = nn.Linear(width, width, device="cpu")
        self.extra = nn.Linear(3, 3, device="cpu")

    def forward(self, x, context):
        # A shape/gradient fixture, not an implementation of native attention.
        key, value = self.to_kv(context).chunk(2, dim=-1)
        return self.to_out(torch.tanh(self.to_q(x) + key + value))


class FakeBlock(nn.Module):
    def __init__(self, active):
        super().__init__()
        self.active = active
        self.cross_attn = FakeCrossAttention(active)
        self.self_attn = nn.Linear(3, 3, device="cpu")
        self.mlp = nn.Linear(3, 3, device="cpu")

    def forward(self, x, context):
        return x + self.cross_attn(x, context) if self.active else x


class FakeNative(nn.Module):
    def __init__(self):
        super().__init__()
        # Unselected blocks have small decoy projections to keep CPU tests light.
        self.blocks = nn.ModuleList(FakeBlock(i in (4, 14, 24)) for i in range(30))
        self.other = nn.Linear(3, 3, device="cpu")

    def forward(self, x, context):
        for block in self.blocks:
            x = block(x, context)
        return x


class StructureLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(8124)
        self.native = FakeNative().eval().requires_grad_(False)
        self.x = torch.randn(2, 3, 1536, device="cpu")
        self.context = torch.randn(2, 3, 1024, device="cpu")

    def wrappers(self):
        return [
            getattr(self.native.blocks[index].cross_attn, name)
            for index in (4, 14, 24) for name in ("to_q", "to_kv", "to_out")
        ]

    def test_only_fixed_nine_projections_and_original_parameter_objects(self):
        modules = dict(self.native.named_modules())
        parameters = dict(self.native.named_parameters())
        self.assertIs(attach_lora(self.native), self.native)
        expected_paths = {
            f"blocks.{index}.cross_attn.{name}"
            for index in (4, 14, 24) for name in ("to_q", "to_kv", "to_out")
        }
        actual_paths = {
            name for name, module in self.native.named_modules()
            if isinstance(module, LoRALinear)
        }
        self.assertEqual(actual_paths, expected_paths)
        self.assertEqual(LORA_BLOCKS, (4, 14, 24))
        self.assertEqual((LORA_RANK, LORA_ALPHA, LORA_DROPOUT), (8, 8, 0.0))
        current_modules = dict(self.native.named_modules())
        for name, module in modules.items():
            if name in expected_paths:
                self.assertIs(current_modules[name].base, module)
            else:
                self.assertIs(current_modules[name], module)
        current_parameters = dict(self.native.named_parameters())
        for name, parameter in parameters.items():
            parent, field = name.rsplit(".", 1)
            current_name = f"{parent}.base.{field}" if parent in expected_paths else name
            self.assertIs(current_parameters[current_name], parameter)
            self.assertFalse(parameter.requires_grad)
        expected_trainable = {
            f"{path}.{field}" for path in expected_paths for field in ("A", "B")
        }
        self.assertEqual(
            {name for name, p in current_parameters.items() if p.requires_grad},
            expected_trainable,
        )
        trainable = list(trainable_parameters(self.native))
        self.assertEqual(len(trainable), 18)
        self.assertEqual(len({id(p) for p in trainable}), 18)
        self.assertEqual({id(p) for p in trainable},
                         {id(current_parameters[name]) for name in expected_trainable})
        self.assertEqual(sum(p.numel() for p in trainable), 245760)
        for layer in self.wrappers():
            self.assertEqual(layer.A.shape, (8, layer.in_features))
            self.assertEqual(layer.B.shape, (layer.out_features, 8))
            self.assertEqual(layer.A.dtype, torch.float32)
            self.assertEqual(layer.B.dtype, torch.float32)
            self.assertTrue(torch.isfinite(layer.A).all())
            self.assertGreater(torch.count_nonzero(layer.A).item(), 0)
            self.assertLessEqual(layer.A.abs().max().item(), 1 / math.sqrt(layer.in_features))
            self.assertEqual(torch.count_nonzero(layer.B).item(), 0)
            self.assertEqual(layer.dropout.p, 0.0)
            self.assertEqual(layer.scaling, 1.0)
            self.assertIs(layer.weight, layer.base.weight)
            self.assertIs(layer.bias, layer.base.bias)
            self.assertFalse(layer.training)

    def test_initial_output_is_strictly_equal_in_train_and_eval(self):
        with torch.no_grad():
            baseline = self.native(self.x, self.context)
        attach_lora(self.native)
        for training in (False, True):
            self.native.train(training)
            with torch.no_grad():
                actual = self.native(self.x, self.context)
            self.assertTrue(torch.equal(actual, baseline))
            for layer in self.wrappers():
                x = self.context if layer.in_features == 1024 else self.x
                self.assertTrue(torch.equal(layer(x), layer.base(x)))

    def test_zero_identity_preserves_original_output_dtype(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for bias in (False, True):
                with self.subTest(dtype=dtype, bias=bias):
                    base = nn.Linear(11, 13, bias=bias, dtype=dtype, device="cpu")
                    base.requires_grad_(False)
                    layer = LoRALinear(base)
                    x = torch.randn(2, 4, 11, device="cpu").to(dtype)
                    actual = layer(x)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(layer.A.dtype, torch.float32)
                    self.assertEqual(layer.B.dtype, torch.float32)
                    self.assertTrue(torch.equal(actual, base(x)))

    def test_fp32_residual_even_inside_cpu_autocast(self):
        base = nn.Linear(11, 13, device="cpu").requires_grad_(False)
        layer = LoRALinear(base)
        with torch.no_grad():
            layer.B.normal_(std=0.2)
        x = torch.randn(2, 4, 11, device="cpu")
        residual = F.linear(F.linear(x.float(), layer.A), layer.B) * (8 / 8)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            base.to(dtype=dtype)
            inputs = x.to(dtype)
            expected_residual = F.linear(F.linear(inputs.float(), layer.A), layer.B)
            actual = layer(inputs)
            self.assertEqual(actual.dtype, dtype)
            self.assertTrue(torch.equal(actual, base(inputs) + expected_residual.to(dtype)))
        base.float()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            baseline = base(x)
            actual = layer(x)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, baseline + residual.to(baseline.dtype)))
        actual.float().square().mean().backward()
        for parameter in (layer.A, layer.B):
            self.assertEqual(parameter.grad.dtype, torch.float32)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().max().item(), 0)

    def test_B_gradients_nonzero_and_optimizer_keeps_base_frozen_unchanged(self):
        originals = [(p, p.detach().clone()) for p in self.native.parameters()]
        attach_lora(self.native)
        optimizer = torch.optim.SGD(trainable_parameters(self.native), lr=0.1)
        output = self.native(self.x, self.context)
        output.square().mean().backward()
        for layer in self.wrappers():
            self.assertIsNotNone(layer.B.grad)
            self.assertTrue(torch.isfinite(layer.B.grad).all())
            self.assertGreater(layer.B.grad.abs().max().item(), 0)
            self.assertIsNotNone(layer.A.grad)
            self.assertTrue(torch.isfinite(layer.A.grad).all())
            self.assertEqual(torch.count_nonzero(layer.A.grad).item(), 0)
        optimizer.step()
        for parameter, original in originals:
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(parameter, original))
        for layer in self.wrappers():
            self.assertGreater(torch.count_nonzero(layer.B).item(), 0)

    def test_frozen_base_still_propagates_input_gradients(self):
        base = nn.Linear(11, 13, device="cpu").requires_grad_(False)
        layer = LoRALinear(base)
        x = torch.randn(2, 11, device="cpu", requires_grad=True)
        expected = torch.autograd.grad(base(x).square().sum(), x)[0]
        actual = torch.autograd.grad(layer(x).square().sum(), x)[0]
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(all(p.grad is None for p in base.parameters()))

    def test_export_import_only_AB_restores_exact_nonzero_result(self):
        attach_lora(self.native)
        with torch.no_grad():
            baseline = self.native(self.x, self.context)
            for layer in self.wrappers():
                layer.B.normal_(std=0.02)
            expected = self.native(self.x, self.context)
        self.assertFalse(torch.equal(expected, baseline))
        state = lora_state_dict(self.native)
        expected_keys = {
            f"blocks.{index}.cross_attn.{name}.{field}"
            for index in (4, 14, 24) for name in ("to_q", "to_kv", "to_out")
            for field in ("A", "B")
        }
        self.assertEqual(set(state), expected_keys)
        for value in state.values():
            self.assertEqual(value.device.type, "cpu")
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)
        # Exercise the usual checkpoint serialization without writing any files.
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        serialized = torch.load(buffer, map_location="cpu", weights_only=True)
        restored = copy.deepcopy(self.native)
        originals = [(layer.base.weight, layer.base.weight.detach().clone())
                     for layer in restored.modules() if isinstance(layer, LoRALinear)]
        with torch.no_grad():
            for parameter in trainable_parameters(restored):
                parameter.zero_()
        self.assertTrue(torch.equal(restored(self.x, self.context), baseline))
        load_lora_state_dict(restored, serialized)
        self.assertTrue(torch.equal(restored(self.x, self.context), expected))
        for parameter, original in originals:
            self.assertTrue(torch.equal(parameter, original))
            self.assertFalse(parameter.requires_grad)
        self.assertTrue(all(p.requires_grad for p in trainable_parameters(restored)))
        # Exported tensors must not alias the live adapter parameters.
        first_key = next(iter(state))
        original = dict(self.native.named_parameters())[first_key].detach().clone()
        state[first_key].add_(1)
        self.assertTrue(torch.equal(dict(self.native.named_parameters())[first_key], original))

    def test_attachment_validates_before_mutating_and_rejects_repeat(self):
        originals = [getattr(self.native.blocks[i].cross_attn, name)
                     for i in (4, 14, 24) for name in PROJECTION_SHAPES]
        self.native.other.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "freeze native BEFORE"):
            attach_lora(self.native)
        self.assertTrue(all(a is b for a, b in zip(self.wrappers(), originals)))
        self.native.requires_grad_(False)
        invalid = nn.Linear(1536, 17, device="cpu").requires_grad_(False)
        self.native.blocks[24].cross_attn.to_out = invalid
        with self.assertRaisesRegex(ValueError, "blocks.24.cross_attn.to_out"):
            attach_lora(self.native)
        self.assertTrue(all(a is b for a, b in zip(self.wrappers()[:-1], originals[:-1])))
        self.native.blocks[24].cross_attn.to_out = originals[-1]
        attach_lora(self.native)
        attached = self.wrappers()
        with self.assertRaisesRegex(ValueError, "already attached"):
            attach_lora(self.native)
        self.assertTrue(all(a is b for a, b in zip(self.wrappers(), attached)))

    def test_load_strict_keys_shapes_and_no_partial_updates(self):
        with self.assertRaisesRegex(ValueError, "attach_lora"):
            lora_state_dict(self.native)
        attach_lora(self.native)
        original = lora_state_dict(self.native)
        changed = {key: value + 1 for key, value in original.items()}
        first_key, last_key = next(iter(changed)), next(reversed(changed))
        missing = dict(changed)
        del missing[first_key]
        invalid_states = (
            missing,
            {**changed, "blocks.4.cross_attn.to_q.base.weight": torch.zeros(1)},
            {**changed, last_key: torch.zeros(1)},
            {**changed, last_key: changed[last_key].to(torch.int64)},
        )
        for invalid in invalid_states:
            with self.subTest(keys=len(invalid)):
                with self.assertRaises(ValueError):
                    load_lora_state_dict(self.native, invalid)
                current = lora_state_dict(self.native)
                self.assertTrue(all(torch.equal(current[key], value)
                                    for key, value in original.items()))


if __name__ == "__main__":
    unittest.main()
