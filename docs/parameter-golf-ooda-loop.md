# Parameter Golf OODA Loop

## Core Reframe

Do not think of this as "a tiny transformer."

Think of it as a repeated control loop over a state:

- the state is the hidden representation
- the reusable blocks are control operators
- the schedule is the order in which those operators act
- the score is not just raw performance, but raw performance after compression

The human question is:

How do we build a repeated control loop whose trajectory is useful before compression and still legible after compression?

## OODA

### Observe

For each experiment, observe only a small fixed set of quantities:

- exact `final_int8_zlib_roundtrip_exact val_bpb`
- compressed artifact bytes
- pass RMS
- relative delta norm
- cosine only as a secondary check
- train speed / runtime damage

These are the ground-truth signals.

### Orient

Interpret the run in signal-language rather than benchmark-language.

There are four useful orientations:

1. `amplify`
   The revisit pushes the current state farther in the same direction.

2. `damp`
   The revisit reduces energy or suppresses higher modes.

3. `mean-revert`
   The revisit pulls the state back toward a stable center or attractor.

4. `least-bad move`
   When unsure, prefer the smallest control that changes exact int8 behavior without adding many new degrees of freedom.

### Decide

Choose one narrow hypothesis at a time.

Good hypotheses look like:

- "decoder revisits should trim overshoot"
- "the second revisit should refine, not amplify"
- "a pivot block should only appear at phase boundaries"

Bad hypotheses look like:

- "make everything stronger"
- "make everything weaker"
- "split every role into many special cases"

### Act

Run the smallest reversible experiment that tests the hypothesis:

- one new scalar family at most
- same `80`-step smoke protocol
- same resonance analysis
- same exact int8 comparison

Kill it quickly if it does not beat or nearly match the current best line.

## Signal Interpretation Of The Current Search

### 1. Amplification

Your note:

- "linear extrapolation cheapest compute possible"

Interpretation:

- revisit amplification is the cheapest way to say "continue moving in this direction"
- mathematically, it is a scalar on the revisit update
- experimentally, we already tested this

What we learned:

- free revisit amplification is real
- it changes the trajectory strongly
- it slightly hurts the exact compressed objective

So amplification matters, but it is too blunt by itself.

### 2. Damping

Your note:

- "we only care about 1st and 2nd modes"

Interpretation:

- do not preserve every mode equally
- preserve the strongest / most stable modes and let weaker modes die
- in architecture terms, revisits should suppress noisy residual detail while retaining large-structure signal

What we learned:

- uniform damping also hurts
- naive phase split hurt even more

So the problem is not simply "less energy." It is selective damping.

### 3. Mean-Reversion Attractor

Your note:

- "mean reversion attractor"

Interpretation:

- the system may want a stable center that revisits return to
- this center might be:
  - the incoming state
  - the original embedding state `x0`
  - a running average
  - a block-specific reference state

Architectural question:

- when a revisit happens, should the block say "go farther" or "come back toward the stable manifold"?

This is likely the next high-value concept, but not with a big bag of knobs.

### 4. Iterate Least-Bad Move

Your note:

- "iterate least bad move"

Interpretation:

- when the mechanism is not yet understood, do not add more freedom
- prefer one tiny asymmetry over many new controls

Operational rule:

- any new hypothesis must earn its keep in one smoke run and one resonance pass

## Compression / Perceptual Coding Hypothesis

Your note:

- "internal dynamics rely on error free calculation final contains entropy"
- "consider ways that perceptual coding wins"

Interpretation:

The trained hidden dynamics may use fine numerical detail that disappears after compression.

Perceptual coding suggests:

- not every error matters equally
- preserve the structure that carries meaning
- let fine-grain detail collapse if it is not functionally important

Architectural translation:

- design revisits to reinforce coarse, stable, reusable structure
- avoid relying on delicate cancellations
- prefer trajectories that remain useful when values are rounded, clipped, and packed

The real objective is not:

- "best hidden-state behavior"

It is:

- "best hidden-state behavior that survives quantization noise"

## Fibonacci / Pivot-Block Reading

Your note:

- "C comes in after n loops of state P"
- "`1, 1, 2 = A, A, C, B...`"

My interpretation:

- `A` and `B` are the two main recurring modes
- `C` is a pivot or recombination block
- `C` should appear when the loop changes scale or phase, not every time

One way to read this is:

- run a mode for a Fibonacci-like number of repeats
- insert `C` when switching modes

Example run-length idea:

- `1` of `A`
- `1` of `A`
- pivot `C`
- `2` of `B`
- pivot `C`
- `3` of `A`

Length-9 example:

- `A A C B B C A A A`

This is not claimed as the only correct reading. It is a codable interpretation of your note.

The key idea is good:

- `C` is not just "third block"
- `C` is the architectural pivot where state recombination or phase change happens

## Architectural Attractors

Your note:

- "what attractors influence at an architectural level?"

The useful architectural attractors are:

1. `carry attractor`
   The residual path wants to keep the state near itself.

2. `origin attractor`
   The system may want to drift back toward the initial embedding state or an early representation.

3. `skip attractor`
   Decoder skip connections pull the state toward previously stored encoder states.

4. `block-role attractor`
   A repeated block tends to settle into a stable behavioral role:
   opener, shaper, refiner, corrector, stabilizer.

5. `schedule attractor`
   A mirror schedule creates a return path, not just a forward path.

6. `compression attractor`
   Some trajectories naturally quantize better:
   narrower ranges, repeated patterns, lower sensitivity to tiny perturbations.

7. `mean-reversion attractor`
   The system may have a stable manifold it wants to return to after each strong move.

The architectural job is to decide which of these attractors should dominate at which revisit.

## The Plain-English Problem

We are not trying to make the model do more things.

We are trying to make it revisit the same small machines in the right way:

- sometimes to push
- sometimes to trim
- sometimes to pivot
- sometimes to come back to center

And we need all of that to remain effective after the model is compressed.
