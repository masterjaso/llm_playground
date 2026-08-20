# p16/top4 high-sparsity research synthesis

The 70.59%-reduction p16/top4 basis is capacity-sufficient but not yet a
trainable production candidate under the full gate. The best FIT/validation
course-correction checkpoint reached validation cosine `0.981599`, NMSE
`0.022887`, dead experts `0`, and load CV `0.442096`. Strict holdout reload
fell to cosine `0.977044` (NMSE `0.027239`, load CV `0.471265`), so the green
gate remains blocked.

The post-selection exact oracle on that same holdout basis reached cosine
`0.982185` (bounded oracle `0.982124`), proving the basis/topology can clear
the gate. The remaining loss is selector generalization: learned-vs-exact
top-k recall was `0.834694` on holdout. Nonlinear routers, exact-set ranking,
soft-load balancing, a hidden-512 router, and a differentiable soft top-k
surrogate did not produce a better validated finalist.

No representative-layer or 64-layer replay was started. Continue only with
FIT/validation development; do not tune against the opened holdout.
