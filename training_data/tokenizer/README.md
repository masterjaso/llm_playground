# Tokenizer samples (v4)

Canonical text remains tokenizer-independent, but production views use the
frozen contract in `production.yaml`
(`gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e`). The release records the
tokenizer fingerprint, vocabulary size, EOS/PAD policy, token format, and
smallest safe integer dtype. Never use a mutable `main` revision for
exact-token accounting.
