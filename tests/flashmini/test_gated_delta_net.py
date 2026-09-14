"""Focused correctness checks for the causal gated delta recurrence."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from flashmini.config import GatedDeltaNetConfig
from flashmini.models.gated_delta_net import GatedDeltaNet


class GatedDeltaNetTests(unittest.TestCase):
    @staticmethod
    def _run_recurrence(
        layer: GatedDeltaNet,
        x: torch.Tensor,
        reference: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, beta, log_decay, gate = layer._project(x)
        recurrence = (
            layer._sequential_recurrence
            if reference
            else layer._chunked_recurrence
        )
        raw, state = recurrence(q, k, v, beta, log_decay)
        output = layer.w_out((gate * raw).to(layer.w_out.weight.dtype)) + layer.norm(x)
        return output, state

    def test_chunked_matches_sequential_outputs_and_gradients(self) -> None:
        config = GatedDeltaNetConfig(
            d_state=4,
            chunk_size=3,
            use_short_conv=True,
            short_conv_kernel=3,
        )
        torch.manual_seed(11)
        chunked = GatedDeltaNet(7, config).double()
        sequential = GatedDeltaNet(7, config).double()
        sequential.load_state_dict(chunked.state_dict())
        x_chunked = torch.randn(2, 8, 7, dtype=torch.float64, requires_grad=True)
        x_sequential = x_chunked.detach().clone().requires_grad_()

        output_chunked, state_chunked = self._run_recurrence(chunked, x_chunked, False)
        output_sequential, state_sequential = self._run_recurrence(
            sequential,
            x_sequential,
            True,
        )
        torch.testing.assert_close(output_chunked, output_sequential, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(state_chunked, state_sequential, rtol=1e-11, atol=1e-11)

        loss_chunked = output_chunked.square().mean() + state_chunked.square().mean()
        loss_sequential = output_sequential.square().mean() + state_sequential.square().mean()
        loss_chunked.backward()
        loss_sequential.backward()
        torch.testing.assert_close(x_chunked.grad, x_sequential.grad, rtol=1e-10, atol=1e-10)
        for (name_chunked, parameter_chunked), (name_sequential, parameter_sequential) in zip(
            chunked.named_parameters(),
            sequential.named_parameters(),
        ):
            self.assertEqual(name_chunked, name_sequential)
            torch.testing.assert_close(
                parameter_chunked.grad,
                parameter_sequential.grad,
                rtol=1e-10,
                atol=1e-10,
                msg=name_chunked,
            )

    def test_suffix_changes_do_not_affect_prefix(self) -> None:
        config = GatedDeltaNetConfig(
            d_state=5,
            chunk_size=4,
            use_short_conv=True,
            short_conv_kernel=4,
        )
        torch.manual_seed(12)
        layer = GatedDeltaNet(8, config).eval()
        prefix_length = 5
        x = torch.randn(2, 11, 8)
        changed = x.clone()
        changed[:, prefix_length:] += 5.0

        with torch.no_grad():
            output = layer(x)
            changed_output = layer(changed)
        torch.testing.assert_close(
            output[:, :prefix_length],
            changed_output[:, :prefix_length],
            rtol=0.0,
            atol=0.0,
        )

    def test_chunk_size_boundaries_match_sequential_reference(self) -> None:
        for chunk_size, steps in ((1, 7), (2, 8), (64, 129)):
            with self.subTest(chunk_size=chunk_size, steps=steps):
                config = GatedDeltaNetConfig(
                    d_state=3,
                    chunk_size=chunk_size,
                    use_short_conv=False,
                )
                torch.manual_seed(20 + chunk_size)
                chunked = GatedDeltaNet(5, config).double()
                sequential = GatedDeltaNet(5, config).double()
                sequential.load_state_dict(chunked.state_dict())
                x_chunked = torch.randn(1, steps, 5, dtype=torch.float64, requires_grad=True)
                x_sequential = x_chunked.detach().clone().requires_grad_()

                output_chunked, state_chunked = self._run_recurrence(chunked, x_chunked, False)
                output_sequential, state_sequential = self._run_recurrence(
                    sequential,
                    x_sequential,
                    True,
                )
                torch.testing.assert_close(
                    output_chunked,
                    output_sequential,
                    rtol=1e-10,
                    atol=1e-10,
                )
                torch.testing.assert_close(
                    state_chunked,
                    state_sequential,
                    rtol=1e-10,
                    atol=1e-10,
                )
                (output_chunked.square().mean() + state_chunked.square().mean()).backward()
                (output_sequential.square().mean() + state_sequential.square().mean()).backward()
                torch.testing.assert_close(
                    x_chunked.grad,
                    x_sequential.grad,
                    rtol=1e-9,
                    atol=1e-10,
                )

    def test_extreme_decay_matches_sequential_outputs_and_gradients(self) -> None:
        config = GatedDeltaNetConfig(d_state=4, chunk_size=64, use_short_conv=False)
        layer = GatedDeltaNet(6, config).double()
        batch, steps, state_dim = 2, 129, config.d_state
        torch.manual_seed(21)
        q_base = torch.randn(batch, steps, state_dim, dtype=torch.float64)
        k_base = torch.randn(batch, steps, state_dim, dtype=torch.float64)
        v_base = torch.randn(batch, steps, state_dim, dtype=torch.float64)
        beta_base = torch.randn(batch, steps, 1, dtype=torch.float64)
        decay_values = torch.tensor(
            [-1000.0, -100.0, -80.0, -1e-6, -0.001, -10.0],
            dtype=torch.float64,
        )
        log_decay_base = decay_values.repeat((batch * steps + decay_values.numel() - 1) // decay_values.numel())[
            : batch * steps
        ].reshape(batch, steps, 1)

        def leaves() -> tuple[torch.Tensor, ...]:
            return tuple(
                value.detach().clone().requires_grad_()
                for value in (q_base, k_base, v_base, beta_base, log_decay_base)
            )

        inputs_chunked = leaves()
        inputs_sequential = leaves()
        q_c, k_c, v_c, beta_c, log_decay_c = inputs_chunked
        q_s, k_s, v_s, beta_s, log_decay_s = inputs_sequential
        raw_chunked, state_chunked = layer._chunked_recurrence(
            F.normalize(q_c, dim=-1),
            F.normalize(k_c, dim=-1),
            v_c,
            torch.sigmoid(beta_c),
            log_decay_c,
        )
        raw_sequential, state_sequential = layer._sequential_recurrence(
            F.normalize(q_s, dim=-1),
            F.normalize(k_s, dim=-1),
            v_s,
            torch.sigmoid(beta_s),
            log_decay_s,
        )
        torch.testing.assert_close(raw_chunked, raw_sequential, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(state_chunked, state_sequential, rtol=1e-10, atol=1e-10)
        (raw_chunked.square().mean() + state_chunked.square().mean()).backward()
        (raw_sequential.square().mean() + state_sequential.square().mean()).backward()
        for chunked_input, sequential_input in zip(inputs_chunked, inputs_sequential):
            torch.testing.assert_close(
                chunked_input.grad,
                sequential_input.grad,
                rtol=1e-9,
                atol=1e-10,
            )

    def test_invalid_chunk_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GatedDeltaNet(4, GatedDeltaNetConfig(d_state=2, chunk_size=0))

    def test_half_input_uses_finite_fp32_recurrence(self) -> None:
        config = GatedDeltaNetConfig(
            d_state=4,
            chunk_size=4,
            use_short_conv=False,
        )
        torch.manual_seed(13)
        layer = GatedDeltaNet(8, config).half().eval()
        x = torch.randn(2, 9, 8, dtype=torch.float16)
        with torch.no_grad():
            output = layer(x)
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for autocast coverage")
    def test_cuda_bfloat16_autocast_backward_is_finite(self) -> None:
        config = GatedDeltaNetConfig(d_state=4, chunk_size=4, use_short_conv=True)
        layer = GatedDeltaNet(8, config).cuda().train()
        x = torch.randn(2, 9, 8, device="cuda", requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = layer(x)
            loss = output.float().square().mean()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in layer.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
