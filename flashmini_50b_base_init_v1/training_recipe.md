# v4 training recipe (implemented by `python -m flashmini.v4_train`)

Loss: `total = 1.0 * main + mtp_coefficient * mean(MTP t+2, t+3, t+4) + router_aux_coefficient * router_aux`.
`mtp_coefficient` is 0.30 while fewer than 70% of the planned tokens have been
consumed and 0.10 afterwards. Every term is logged separately with the total.

MTP uses ground-truth teacher forcing: depth k at position t embeds `x_(t+k)` and
predicts `x_(t+k+1)`; no sampled rollout.

Router balancing: Qwen global-batch load-balancing `E * sum_i f_i P_i` per logical
MoE layer, averaged over layers; expert counts are all-reduced across data-parallel
ranks and accumulated over gradient-accumulation microbatches
(`exact_prepass` gives the exact logical-batch gradient; `ga_buffer` is Qwen's buffer).
The coefficient has no default; startup fails until it is set.

Optimizers: Muon (Newton-Schulz, 8 steps, per logical operator slice) for hidden
matrices; AdamW for embeddings, LM head, router/gates/HC control matrices (with
explicit weight decay) and norms/GDN scalars/convs (no decay); row-sparse Adam
(weight decay 0) for the host-resident PLE tables. Hyperparameters are donor inputs.

Donor inputs that must be set explicitly (startup fails on null): nodes/GPUs,
micro-batch, gradient accumulation, curriculum phases (sequence length, token
budget, data manifest), teacher mixture, planned total tokens, learning rates and
schedule, router coefficient and balance mode. See `train_example.yaml`.
