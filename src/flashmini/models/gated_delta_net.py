"""Causal gated delta-rule linear attention.

The recurrent state is a ``d_state x d_state`` matrix.  At position ``t`` the
layer applies a scalar decay and a delta write:

``S_t = a_t S_{t-1} + beta_t k_t (v_t - a_t k_t S_{t-1})^T``

where ``q_t`` and ``k_t`` are L2-normalized.  The chunk implementation solves
the resulting lower-triangular system, rather than materializing an attention
mask that can accidentally include future positions.  Its state and solves
run in fp32 for half/bfloat16 inputs to keep long cumulative decays stable.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import nn

from ..config import GatedDeltaNetConfig


class GatedDeltaNet(nn.Module):
    """Gated delta-rule linear-attention layer with an exact chunked scan."""

    def __init__(self, d_model: int, config: GatedDeltaNetConfig):
        super().__init__()
        self.d_model = d_model
        self.config = config
        d = d_model
        ds = config.d_state
        if config.chunk_size <= 0:
            raise ValueError("GatedDeltaNet chunk_size must be positive")

        self.w_gate = nn.Linear(d, ds, bias=False)
        self.w_q = nn.Linear(d, ds, bias=False)
        self.w_k = nn.Linear(d, ds, bias=False)
        self.w_v = nn.Linear(d, ds, bias=False)
        # beta controls the strength of the delta write.
        self.w_beta = nn.Linear(d, 1, bias=False)
        # a is an independent scalar state decay.
        self.w_decay = nn.Linear(d, 1, bias=False)
        self.w_out = nn.Linear(ds, d, bias=False)
        self.norm = nn.LayerNorm(d)

        if config.use_short_conv:
            # Causal left padding is applied in forward.  The previous
            # symmetric-padding/trim implementation was also causal, but
            # explicit left padding makes that invariant local and obvious.
            self.short_conv = nn.Conv1d(
                d,
                ds,
                kernel_size=config.short_conv_kernel,
                padding=0,
                groups=1,
            )
        else:
            self.short_conv = None

    @staticmethod
    def _work_dtype(dtype: torch.dtype) -> torch.dtype:
        """Use fp32 recurrence arithmetic for half and bfloat16 inputs."""

        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project inputs and prepare numerically stable recurrence scalars."""

        work_dtype = self._work_dtype(x.dtype)
        gate = torch.sigmoid(self.w_gate(x).to(work_dtype))
        beta = torch.sigmoid(self.w_beta(x).to(work_dtype))
        # log-sigmoid avoids forming ``1 - sigmoid`` when a is close to one.
        log_decay = F.logsigmoid(self.w_decay(x).to(work_dtype))
        q = self.w_q(x).to(work_dtype)
        k = self.w_k(x).to(work_dtype)
        v = self.w_v(x).to(work_dtype)

        if self.short_conv is not None:
            kernel = self.short_conv.kernel_size[0]
            xc = F.pad(x.transpose(1, 2), (kernel - 1, 0))
            xc = self.short_conv(xc).transpose(1, 2)
            k = k + xc[..., : self.config.d_state].to(work_dtype)

        # Normalized keys make the triangular solve well-conditioned at the
        # default d_state/chunk_size while keeping q and k on the same scale.
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
        return q, k, v, beta, log_decay, gate

    @staticmethod
    def _decay_matrices(log_decay: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return cumulative and pairwise decay factors for one chunk.

        ``log_decay`` has shape ``(B, C, 1)``.  The mask is applied before
        exponentiation, and only lower-triangular log ratios are clamped above
        zero.  This prevents a round-off-induced upper-triangular future leak.
        """

        log_cum = torch.cumsum(log_decay.squeeze(-1), dim=1)
        positions = log_cum.unsqueeze(-1) - log_cum.unsqueeze(-2)
        lower = torch.ones(
            log_cum.shape[-1],
            log_cum.shape[-1],
            device=log_cum.device,
            dtype=torch.bool,
        ).tril()
        # Upper entries are -inf before exp, so they are exactly zero and can
        # never contribute a future token even when ratios round upward.
        masked_log_decay = torch.where(
            lower,
            positions.clamp(max=0.0),
            torch.full_like(positions, -torch.inf),
        )
        pair_decay = torch.exp(masked_log_decay)
        cumulative = torch.exp(log_cum.clamp(max=0.0)).unsqueeze(-1)
        to_end = torch.exp((log_cum[:, -1:] - log_cum).clamp(max=0.0)).unsqueeze(-1)
        return cumulative, pair_decay, to_end

    def _sequential_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference recurrence used for focused equivalence checks.

        This is deliberately a direct token-by-token implementation of the
        delta rule.  It returns both outputs and the state after the sequence.
        """

        batch, steps, state_dim = q.shape
        state = torch.zeros(batch, state_dim, state_dim, device=q.device, dtype=q.dtype)
        outputs: list[torch.Tensor] = []
        for t in range(steps):
            a_t = torch.exp(log_decay[:, t])
            beta_t = beta[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            previous_key_value = torch.bmm(k_t.unsqueeze(1), state).squeeze(1)
            residual = v_t - a_t * previous_key_value
            state = (
                a_t[:, :, None] * state
                + beta_t[:, :, None] * k_t[:, :, None] * residual[:, None, :]
            )
            outputs.append(torch.bmm(q[:, t : t + 1], state).squeeze(1))

        if not outputs:
            return q.new_empty(batch, 0, state_dim), state
        return torch.stack(outputs, dim=1), state

    def _chunked_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the exact delta recurrence in practical-size chunks."""

        batch, steps, state_dim = q.shape
        chunk_size = self.config.chunk_size
        if steps == 0:
            state = torch.zeros(batch, state_dim, state_dim, device=q.device, dtype=q.dtype)
            return q.new_empty(batch, 0, state_dim), state
        pad = (chunk_size - steps % chunk_size) % chunk_size
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            # A padded position has no write and no decay.  Its q/k/v are zero.
            beta = F.pad(beta, (0, 0, 0, pad), value=0.0)
            log_decay = F.pad(log_decay, (0, 0, 0, pad), value=0.0)

        padded_steps = q.shape[1]
        chunks = padded_steps // chunk_size
        q = q.reshape(batch, chunks, chunk_size, state_dim)
        k = k.reshape(batch, chunks, chunk_size, state_dim)
        v = v.reshape(batch, chunks, chunk_size, state_dim)
        beta = beta.reshape(batch, chunks, chunk_size, 1)
        log_decay = log_decay.reshape(batch, chunks, chunk_size, 1)

        state = torch.zeros(batch, state_dim, state_dim, device=q.device, dtype=q.dtype)
        outputs: list[torch.Tensor] = []
        eye = torch.eye(chunk_size, device=q.device, dtype=q.dtype)

        for chunk_index in range(chunks):
            qc = q[:, chunk_index]
            kc = k[:, chunk_index]
            vc = v[:, chunk_index]
            beta_c = beta[:, chunk_index]
            log_decay_c = log_decay[:, chunk_index]
            cumulative, pair_decay, to_end = self._decay_matrices(log_decay_c)

            # U_i = beta_i (v_i - a_i k_i S_prev) corrected for all earlier
            # writes in the same chunk.  The coefficient matrix is unit lower
            # triangular, so this solve is exactly the serial delta update in
            # matrix form while remaining causal by construction.
            previous_key_value = torch.bmm(kc, state)
            rhs = beta_c * (vc - cumulative * previous_key_value)
            gram = torch.bmm(kc, kc.transpose(1, 2))
            coefficient = eye.unsqueeze(0) + torch.tril(
                beta_c * pair_decay * gram,
                diagonal=-1,
            )
            updates = torch.linalg.solve_triangular(
                coefficient,
                rhs,
                upper=False,
                unitriangular=True,
            )

            qk = torch.bmm(qc, kc.transpose(1, 2))
            outputs.append(
                cumulative * torch.bmm(qc, state)
                + torch.bmm(torch.tril(pair_decay * qk), updates)
            )

            state = cumulative[:, -1:, :] * state + torch.bmm(
                kc.transpose(1, 2),
                to_end * updates,
            )

        outputs_tensor = torch.cat(outputs, dim=1)
        return outputs_tensor[:, :steps], state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, steps, _ = x.shape
        if steps == 0:
            return self.norm(x)

        q, k, v, beta, log_decay, gate = self._project(x)
        # CUDA/CPU autocast can otherwise downcast bmm/solve even when the
        # projected tensors were explicitly promoted to fp32.  Keep the full
        # recurrence in its work dtype (fp32 for AMP inputs) until w_out.
        recurrence_context = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if x.device.type in ("cuda", "cpu")
            else contextlib.nullcontext()
        )
        with recurrence_context:
            recurrent, _ = self._chunked_recurrence(q, k, v, beta, log_decay)
            output = gate * recurrent
        # Explicitly cast back after fp32 recurrence so the projection follows
        # the module/input dtype under ordinary execution and autocast.
        output = self.w_out(output.to(self.w_out.weight.dtype))
        return output + self.norm(x)
