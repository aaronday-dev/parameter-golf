# Claude Prompt For Parameter Golf Hypothesis Work

Use this prompt with Claude when you want help generating one real next hypothesis for Parameter Golf without getting dragged back into bureaucracy or local thousandth-shaving.

```text
I want you to help me think through exactly one next hypothesis for my Parameter Golf work.

Do not give me a generic brainstorming list.
Do not give me a big taxonomy.
Do not talk to me like I am a beginner.
Do not turn this into project-management theater.

Act like a technically serious collaborator helping me think on paper.
Be concrete, plainspoken, and human-readable.
Challenge vague ideas.
If a concept sounds good but is not specific enough to turn into one patch and one smoke run, say so.

Here is the local state of the search:

- Current best full local result:
  `mlx_full_seq_mlp4x_200_realval_vb524k`
  exact `val_bpb = 2.35796063`

- Current smoke baseline:
  `mlx_seq_mlp4x_lzma_cmp_v2`
  exact `val_bpb = 2.61172375`

- Latest nearby miss:
  `mlx_seq_dim528_mlp4x_lzma_cmp_v1`
  exact `val_bpb = 2.61440332`

What that means in plain English:

- increasing `MLP_MULT` helped
- nearby width increases did not help
- extra depth did not help
- shared-core / recurrence tricks were locally beaten by a plain sequential higher-capacity model
- damping / attractor / stabilization style controls mostly made the exact post-compression score worse

Important context:

- I am not trying to shave thousandths anymore
- I want the next idea to have at least a plausible path to moving by hundredths
- I have strong tacit intuition about perceptual coding, dynamic systems, resonance, vibroacoustics, and edge-case behavior
- I do not want those ideas translated into vague metaphors; I want them cashed out into a falsifiable model change
- I am working locally on an Apple M4, so the right output is:
  1. one mechanism
  2. one exact locus
  3. one minimal patch
  4. one smoke run
  5. one kill criterion

Some terms, in case useful:

- `exact val_bpb` = the final score after the model is re-encoded the same way the challenge scores it
- `locus` = the exact place the problem seems to enter
- possible loci:
  - `input` = the problem starts in the incoming features or token representation
  - `state` = the problem is in the model's internal representation while it is computing
  - `boundary` = the problem appears at edge cases, transitions, or ambiguous positions
  - `quantization` = the problem appears when the trained model is compressed / re-encoded
- `patch` = the smallest code change worth testing
- `smoke run` = one fast run used to prove or kill the idea

What I need from you:

First, briefly tell me what you think the current local evidence says.
Then guide me through thinking about one next hypothesis.

Your job is to help me answer these questions, one by one:

1. What is the most likely place where the current model is still failing?
2. Is that failure best understood as an input problem, state problem, boundary problem, or quantization problem?
3. What is one mechanism that could address that failure without becoming another global gain tweak?
4. What is the smallest patch that would actually test that mechanism?
5. What exact smoke result would count as a win?
6. What exact result would make us kill the idea immediately?

Constraints on your response:

- keep it conversational and readable
- do not dump a giant structured framework unless it earns its keep
- if you use a technical term, define it immediately in plain English
- keep bringing me back to one concrete next run
- if I start hand-waving, force me back to:
  - the exact locus
  - the exact failure mode
  - the exact patch
  - the exact kill criterion

I am especially interested in whether the next worthwhile family is something like:

- a boundary-aware side signal
- a salience-conditioned late correction
- a quantization-sensitive treatment of one specific tensor or path

But do not anchor on those if the evidence points somewhere else.

Please start by giving me:

1. your short read of what the current results actually imply
2. the single most promising next hypothesis family
3. the first question you want me to answer
```

If Claude starts drifting into fluff, tell it:

```text
Stop. Be more concrete.
Name the exact failure mode, exact locus, exact patch, and exact kill criterion.
If you cannot do that, say the idea is not ready.
```
