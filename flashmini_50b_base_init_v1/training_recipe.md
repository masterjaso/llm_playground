# Training recipe boundary

This is an initialization handoff, not a training recipe. Donor-side decisions intentionally remain open: global token batch, peak learning rate, teacher-data percentage, long-context curriculum, and optimizer hyperparameters other than the frozen Muon/Adam taxonomy. MTP auxiliary coefficient is 0.30 for the first approximately 70% of training and 0.10 for the final approximately 30%; the main loss weight is 1.0. Canonical training uses ground-truth shifted embeddings and no sampled rollout.
