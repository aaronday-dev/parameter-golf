# Parameter Golf Next Runs

This note captures the branching logic that followed the promoted
`MLP_MULT=3` shared-core run.

Historical outcome:

- Branch A regressed on smoke at `2.66895938`
- Branch B promoted cleanly and became the current local leader at `2.37334218`

## Branch A: Width-First On The Winning Shared-Core Family

Use this if the current promoted `mirror + dirc02 + MLP_MULT=3` run is
competitive enough that the mechanism still looks alive.

Smoke:

```bash
RUN_ID=mlx_mirror_dim640_mlp3x_dirc02_cmp \
SHARED_CORE_BLOCKS=3 \
SHARED_CORE_SCHEDULE=mirror \
SHARED_CORE_ROLE_GAINS=0.950,1.050,1.000 \
SHARED_CORE_DIRECTIONAL_CORRECT=1 \
SHARED_CORE_DIRECTIONAL_CORRECT_BLOCK=C \
SHARED_CORE_DIRECTIONAL_CORRECT_SOURCES=AB \
SHARED_CORE_DIRECTIONAL_CORRECT_REVISIT_ONLY=1 \
SHARED_CORE_DIRECTIONAL_CORRECT_MAX_GAIN=0.020 \
SHARED_CORE_DIRECTIONAL_CORRECT_TARGET_AMP=1.100 \
SHARED_CORE_DIRECTIONAL_CORRECT_LOG_BAND=0.350 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=5 \
MLP_MULT=3 \
WARMUP_STEPS=0 \
ITERATIONS=80 \
scripts/run_parameter_golf_mlx_smoke.sh
```

Kill criteria:

- worse than `2.64687904` exact smoke
- clearly slower without a score gain
- compressed artifact growth without predictive benefit

## Branch B: Sequential Capacity Challenger

Use this if the current shared-core family misses the promoted target or if we
want to challenge the shared-core premise directly.

This branch spends more of the `16 MB` budget on plain sequential unique-block
capacity and removes the directional-correction seam entirely.

Smoke:

```bash
RUN_ID=mlx_seq_mlp3x_cmp \
SHARED_CORE_BLOCKS=9 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
WARMUP_STEPS=0 \
ITERATIONS=80 \
scripts/run_parameter_golf_mlx_smoke.sh
```

Why this branch exists:

- old local evidence in `mlx_smoke.txt` shows a much larger sequential MLX model
  can still fit comfortably under the artifact cap
- the current search may be over-indexing on shared-core recurrence and
  under-indexing on straightforward capacity that survives quantization

Kill criteria:

- cannot stay under the artifact cap trajectory
- fails to beat the current strong smoke branch family
- runtime explodes without clear predictive gain
