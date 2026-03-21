# Parameter Golf Thinking Sheet

## What Problem Are We Actually Solving?

We are trying to build a very small language model that performs well under a tight size limit.

The current best idea is not "make nine different layers."
It is "make three reusable computational blocks and choose the order and strength with which we revisit them."

For this sheet, call the three reusable blocks:

- `A`
- `B`
- `C`

The two important schedules so far are:

- cyclic: `A B C A B C A B C`
- mirror: `A B C B A B C B A`

The mirror schedule has been better than the cyclic schedule in our smoke experiments.

## Translation Table

`block`
: one reusable module inside the model

`block identity`
: which module it is, `A`, `B`, or `C`

`revisit`
: using a block again after it has already appeared once

`revisit count`
: whether this is the second, third, or fourth time that block has appeared

`revisit gain`
: how hard that reused block pushes the state when we come back to it

`quantization`
: the final model is compressed; what matters is not only whether the model learns, but whether the learned structure survives compression

## What We Know Right Now

- `mirror` beats `cyclic`
- free revisit amplification changes the dynamics a lot, but did not improve the exact post-compression score
- damping revisits made the score worse
- splitting revisit strength into encoder vs decoder made the score much worse

That means:

- repeated use matters
- order matters
- revisit strength matters
- but the useful region is narrow

## The Human Question

The machine has narrowed the problem to this:

When a small set of reusable blocks is revisited, what is the right job of the revisit?

Possible answers:

- refine what is already there
- correct an error
- amplify a useful mode
- stabilize the trajectory

The machine experiments suggest that "push harder" and "push softer everywhere" are both too crude.

## Exercises

### 1. Draw The Two Paths

Write the two schedules by hand and mark revisits.

- cyclic: `A B C A B C A B C`
- mirror: `A B C B A B C B A`

Questions:

- In which positions does the first real difference appear?
- Which block gets revisited most often in the mirror schedule?
- Which schedule feels more like "go out and come back"?

### 2. Assign Roles To A, B, C

Invent a story for the three blocks.

Example role types:

- opener
- shaper
- confirmer
- corrector
- resonator
- stabilizer

Write one sentence for each:

- `A` is the block that ...
- `B` is the block that ...
- `C` is the block that ...

Then ask:

- does the mirror schedule make more sense under those roles than the cyclic schedule?

### 3. Revisit Count Table

Fill this out for the mirror schedule `A B C B A B C B A`.

| Position | Block | First visit or revisit? | Revisit count |
| --- | --- | --- | --- |
| 1 | A |  |  |
| 2 | B |  |  |
| 3 | C |  |  |
| 4 | B |  |  |
| 5 | A |  |  |
| 6 | B |  |  |
| 7 | C |  |  |
| 8 | B |  |  |
| 9 | A |  |  |

Then answer:

- Should the second visit to `B` have the same job as the fourth visit to `B`?

### 4. Think Like A Signal Designer

Pretend this is a signal chain, not a language model.

What would each of these mean?

- revisit as refinement
- revisit as correction
- revisit as amplification
- revisit as stabilization

For each one, write:

- what the block is trying to do to the signal
- what too much of it would sound like
- what too little of it would sound like

### 5. Compression Question

The game scores the compressed model, not only the raw trained model.

Write three guesses for why a change could make the internal dynamics look stronger while making the final compressed score worse.

Prompt:

- "This change helped during learning, but hurt after compression because ..."

### 6. Fibonacci Exercise

Now try a Fibonacci-style schedule idea.

Start with:

- `F1 = A`
- `F2 = B`

Then build:

- `F3 = B A`
- `F4 = B A B`
- `F5 = B A B B A`

That is one simple Fibonacci-like growth rule: each new word is the previous word followed by the one before it.

Now do three things:

1. Write the first `9` positions of a Fibonacci-style reuse word using only `A` and `B`.
2. Decide how `C` enters the story.
3. Propose one `A/B/C` schedule of length `9` that feels Fibonacci-like rather than cyclic or mirrored.

Questions:

- Does Fibonacci growth create a useful pattern of revisits?
- Does it create too much bias toward one block?
- Would it be better as a two-block schedule with `C` used only as a reset or pivot?

### 7. Final Synthesis

Write one paragraph answering:

"If revisits are neither supposed to be uniformly stronger nor uniformly weaker, then what are they supposed to be?"

Keep it concrete.

Good answers will mention:

- which block
- which revisit number
- whether the job is refinement, correction, amplification, or stabilization
- what compression might punish

## If You Want One Sentence To Sit With

The problem is not "how do I make the model deeper?"
The problem is "how do I reuse a few little machines so that coming back to them helps the trajectory instead of distorting it?"
