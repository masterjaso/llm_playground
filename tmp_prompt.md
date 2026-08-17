Plan
Qwen3.8-27B Dense-to-MoE Completion Plan
Summary
Continue from clean, pushed head ce9740d on agent/windows-dense2moe-real-pipeline, using native Windows exclusively for scientific runs.
Treat Qwen alignment as functional preservation against the pinned dense teacher: NMSE, cosine, output amplitude, and whole-model distribution.
Retain both fixed profiles:
p16/top4: active width 5120, FFN reduction 70.5882%.
p32/top5: active width 3584, FFN reduction 79.4118%.
Current Phase 01 is provenance-green but scientifically YELLOW: p16 oracle NMSE 0.03920, cosine 0.97914, loadCV 0.65825; its learned selector remains unresolved.
Candidate promotion requires independent, never-trained-on generalization corpora. Passing the development corpus alone can never make a candidate green.
Implementation and Execution
1. Re-baseline and close foundation gaps
Populate and validate the NSP repository fact ledger to DISCOVERY_READY; refresh the stale epic/Ralph state to current Windows evidence and head.
Replace placeholder Phase 02–09 commands with real profile-aware gates. Remove p16-hardcoding from representative, full64, assembly, evaluation, and runtime phases.
Implement true sparse gather/dispatch/scatter execution. The current MoE computes every expert and masks outputs; the replacement must invoke only top-k experts per token while computing the shared branch once.
Add dispatch receipts proving per-token expert count, active intermediate width, estimated FFN MAC reduction, absence of dense/all-expert fallback, and Windows runtime identity.
Build a real Qwen3.5 full-model class by reusing the installed attention and Gated DeltaNet implementation and replacing only all 64 dense SwiGLU FFNs.
Require strict fresh-process reload, exact non-FFN tensor preservation, and deterministic logit parity.
Commit these fixes, obtain a clean tree, rerun Windows qualification, and issue a new runtime lock tied to the exact scientific commit.
2. Establish an anti-overfitting data protocol
Create three independently frozen data tiers before primary training:
Development Corpus V2.2: FIT-TRAIN and FIT-DEV, used for optimization and candidate selection.
Internal promotion corpus: GATE-A, SHADOW-B, and SHADOW-C, never used for optimizer updates.
External generalization corpora G1 and G2: independently acquired after the training recipe and thresholds are locked; never used for hyperparameter, partition, router, checkpoint, or threshold selection.
Split by task, repository, trajectory, document lineage, and source family—not random token rows—so related material cannot cross tiers.
Enforce exact normalized-content hashes, near-duplicate/MinHash checks, source-record identity checks, benchmark exclusion, and repository/task disjointness.
Freeze all thresholds, metrics, seeds, and promotion rules before opening GATE-A, SHADOW-B/C, G1, or G2.
Opening an evaluation tier is one-way:
A failure may be diagnosed, but that tier becomes retired for future promotion.
Any method changed because of its results must receive a new method version and pass a newly acquired untouched external corpus.
No repeated tuning against G1/G2 is permitted.
Promotion requires agreement across domains and corpora, not only acceptable global averages. Report code, agentic workflows, technical prose, structured data, general language, hard-residual quartile, and sequence-length slices separately.
3. Freeze production Corpus V2.2
Acquire at least 96 independent, non-benchmark software-agent tasks with pinned revisions, licenses, visible transcripts only, and no private reasoning.
Freeze 750k balanced activation tokens using the existing target mix: 44% code, 28% agentic software engineering, 12% software-engineering prose, 10% general, and 6% structured data.
Produce immutable split identities with tokenizer, overlap, benchmark-exclusion, source, grouping, and hash receipts.
Keep the historical holdout permanently closed.
Capture real layer-0 Qwen X/Y activations, beginning with the contracted 128k FIT-TRAIN production run and fixed evaluation captures. Large tensors remain uncommitted and sharded/mmap-backed.
4. Select p16/top4 and p32/top5 development candidates
Screen source-derived partitions from activation_magnitude, output_contribution, balanced_signature, and residual_swap_refined.
For p16/top4, use exhaustive C(16,4)=1820 projected-positive oracle assignments and a load-priced Pareto frontier.
For p32/top5, use explicitly bounded correlation-ranked pools, expanding until quality and route selections stabilize; never present bounded evidence as exhaustive.
Apply this funnel independently:
Train/refine the basis on FIT-TRAIN.
Rank oracle ceilings on FIT-DEV.
Retain the top two quality/load Pareto candidates.
Freeze each basis and train selection plus positive-amplitude routers with three fixed seeds.
Permit one predeclared joint-refinement fallback only when diagnostics prove a basis limitation.
Freeze the winning development checkpoint and all decisions before opening promotion data.
Development success is only DEV_FINALIST; it is never product-green.
5. Independent promotion and generalization
Evaluate frozen finalists sequentially on GATE-A, SHADOW-B, and SHADOW-C without further fitting.
A candidate that survives internal promotion is evaluated on independently acquired G1 and G2.
Promotion requires on every untouched corpus:
NMSE <= 0.05.
cosine >= 0.98.
loadCV <= 0.50.
zero dead experts.
oracle regret <= 0.10.
repeated-run variation <= 5%.
median ||student||₂ / ||teacher||₂ within 0.95–1.05.
p95 absolute relative output-norm error <= 15%.
Require each major domain slice to remain at least YELLOW and the global result to be GREEN; a strong majority domain cannot hide a failed agentic/code slice.
Report development-to-external degradation explicitly. A material collapse on G1 or G2 rejects the candidate even if its absolute aggregate narrowly passes.
Emit immutable receipts for both topologies, including rejected candidates and every opened evaluation identity.
6. Representative transfer and winner choice
Freeze separate p16 and p32 method locks and evaluate both on layers 0–3, 28–31, and 60–63.
Use the locked seed on all 12 layers and two additional seeds on sentinel layers 3, 31, and 63.
Evaluate representative transfer on development data and untouched external generalization data; do not assume layer-0 generalization transfers to later layers.
Reject a topology after any persistent quality, load, amplitude, generalization, reload, or provenance failure.
Winner rule:
Choose p32/top5 when it is green across the representative matrix and external corpora.
Otherwise choose p16/top4 if it is green.
If neither is green, stop with no product candidate rather than relaxing thresholds.
7. Full64 model and whole-model validation
Run only one active full64 build at a time, streaming captures and layer training with resumable, hash-addressed checkpoints.
Assemble the winner into a BF16 Hugging Face checkpoint while preserving the tokenizer, chat template, embeddings, attention/Gated DeltaNet, norms, and language head.
Require all 64 layer gates plus strict inventory/reload, sparse-dispatch, and external-generalization proofs.
Whole-model validation must include both the fixed internal suite and a fresh post-assembly corpus not used during layer selection.
Whole-model green requires:
perplexity increase <= 5%;
mean token KL <= 0.10;
teacher top-1 agreement >= 85%;
no failed critical domain slice;
coding-agent and preservation-canary reports with no benchmark contamination.
If p32 fails, allow one bounded repair. Because the failure corpus is then contaminated, the repaired version must pass a new untouched external corpus. If it still fails, activate the frozen p16 fallback recipe.
8. HF/PyTorch and GGUF runtime completion
Make HF/PyTorch BF16 sparse execution a blocking completion gate, including equivalence between masked-reference and sparse-dispatch paths.
Pin a tested upstream llama.cpp revision. Extend its Qwen35MoE support for the separate selection/amplitude routers and always-on shared branch.
Extend GGUF export to a complete tensor/name/metadata mapping and maintain a pinned llama.cpp patch series.
Require native-Windows llama.cpp loading, sparse-dispatch evidence, and BF16/F16 HF-to-GGUF parity.
Build an expert-covering imatrix and conservative Q8-like candidate.
Quantized completion requires the model to remain inside the overall dense-teacher quality and external-generalization envelope; report BF16-to-quantized degradation separately.
Public Contracts
Add versioned receipts for:
corpus grouping, disjointness, near-duplicate exclusion, and tier-opening history;
candidate evaluation and external-generalization results;
output amplitude and sparse dispatch;
full-model assembly and whole-model comparison;
GGUF runtime, imatrix coverage, and quantization.
Candidate receipts must bind topology, partition, source revision, dataset and split hashes, method version, code commit, runtime lock, seeds, checkpoints, opened evaluation tiers, metrics, and promotion decision.
Add a contamination ledger that makes reuse of an opened GATE/SHADOW/external corpus by a modified method fail closed.
Replace placeholder CLI behavior with fail-closed commands for corpus freeze, candidate search, independent promotion, representative transfer, full64 assembly, whole-model comparison, GGUF validation, quantization, and benchmarking.
Test and Closeout Plan
Preserve the current 63/63 focused-test baseline.
Add tests for grouped splitting, cross-tier source leakage, exact and near duplicates, evaluation-tier retirement, method-version changes, and rejection of tuning against opened external data.
Add sparse-dispatch equivalence, gradients, empty-expert, dispatch-count, and unselected-expert tests.
Add candidate-gate tests covering domain slices, development-to-external regression, bounded-p32 labeling, seed variation, amplitude, and sealed-split enforcement.
Add full Qwen replacement, non-FFN preservation, 64-layer replacement, fresh-process reload, HF/GGUF parity, llama.cpp runtime, imatrix, and quantized-quality tests.
Keep code commits separate from scientific evidence commits. Run decisive experiments only from clean commits with matching Windows runtime locks.
After all gates pass, run NSP maintenance/evidence closeout, commit final receipts and documentation, push the branch, and report the final SHA, both candidate decisions, generalization gaps, winner, runtime evidence, and reproduction commands.
Assumptions
Source remains Qwen/Qwen3.8-27B at revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
p32/top4 remains inactive.
No historical holdout, replay continuation, evaluation-data tuning, or threshold relaxation is permitted.
“Aligned to Qwen” means functionally aligned under dense-teacher, amplitude, cross-corpus, and whole-model gates.
Both HF/PyTorch and GGUF sparse execution are required before completion.
A candidate is not green unless it generalizes to multiple independently acquired corpora it could not have influenced.
