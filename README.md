# llama-anatomy — a 100k-parameter LLaMA decoder, with every one of its four choices measured against its own control

**llama-anatomy** implements the LLaMA block from the papers (Touvron et al.,
arXiv [2302.13971](https://arxiv.org/abs/2302.13971) and
[2305.11276](https://arxiv.org/abs/2305.11276)) — RMSNorm pre-norm, RoPE, a
SwiGLU feed-forward, grouped-query attention, no biases anywhere, embeddings
tied below 1B — and then runs the experiment the paper cannot: it replaces each
choice, one at a time, with the alternative it was adopted against, and trains
all of them on tasks with exact verifiers. 102,528 parameters, CPU only, three
seeds per row, no GPU and no API key.

![ci](https://github.com/Lesereingrape/llama-anatomy/actions/workflows/ci.yml/badge.svg)

## Why run this at 10^5 parameters

LLaMA's component evidence is a scaling sweep at 7B–65B. Nobody can re-run that,
which means the usual explanations of *why* rotary positions or gated
feed-forwards are standard practice are assertions most of us have never been
able to check. At a fifth of a million parameters the whole ablation matrix,
rotation sweep included, is a half-hour CPU job, so the claims can be re-derived,
and the interesting question stops being "does this scale" and becomes "what does
this mechanism actually do" — measurable at the only scale you can afford several
seeds at.

That framing also decides what this repository is *not*: it is not a language
model, and a mechanism that is free at 10^5 parameters can still be decisive at
10^9. Every result below is labelled with the exact budget it was measured
under, and the conclusions are written at that scale and no larger.

## The setup

- **One config, six rows.** `LLAMA` is a frozen dataclass and every control is
  that config with exactly one field replaced (`VARIANTS` in
  `src/llamanat/model.py`), so two rows of any table differ in one mechanism
  and nothing else: `layernorm` swaps RMSNorm for `nn.LayerNorm`, `abs_pos`
  swaps RoPE for a learned position table, `gelu_mlp` swaps SwiGLU for a plain
  GeLU MLP, `mha`/`mqa` move only the number of KV heads.
- **Equal-parameter controls.** SwiGLU spends three `d×ff` matrices where a
  plain MLP spends two, so the GeLU control runs at `hidden = 1.5 * ff`
  (`blocks.gelu_hidden`) and lands on exactly the same parameter count. Without
  that, the gating ablation would be a size ablation.
- **One budget, one initialisation.** Same AdamW loop, same batch, same step
  count, same `normal(0, 0.02)` init with residual output projections scaled by
  `1/sqrt(2*n_layer)`. A variant never loses because it was tuned harder.
- **And one axis that costs nothing.** RoPE arrives with a single number
  attached, the base of the frequency series, and changing it adds no parameter,
  no bias and no shape: the swept models are the same 102,528 weights with
  different angles. That makes the sweep in `How fast the rotation runs` the one
  place where a difference cannot be a size difference, and the reach column
  beside it is arithmetic on the geometry rather than a measurement.
- **Curves, not snapshots.** Each row is *one* training trajectory scored at
  checkpoints 300/600/900/1200, because these tasks are learned by phase
  transition: a single fixed step records where a variant happens to sit
  relative to its own threshold, not what it is capable of.
- **Two verifiable tasks** (`src/llamanat/tasks.py`). *Addition*: three-digit
  sums emitted as a digit/carry chain, least-significant column first so the
  recurrence is causal; graded by reconstructing the number after free-running
  generation — the grader never trusts the model's own arithmetic. *Recall*: `n`
  key/value pairs, then a probe key; graded by whether the emitted symbol is the
  one that value was *named* with. Neither task pads: every sequence in a batch
  has the same length, so a shape bug cannot hide behind a pad token.
- **Train short, test long** (cf. Press et al., arXiv
  [2108.12409](https://arxiv.org/abs/2108.12409)): the same fitted weights are
  re-scored at 8 and 12 pairs after training on 4. The learned table is sized to
  the evaluation length on purpose, so out-of-range rows *exist* — they were
  simply never updated. That is the comparison, not a handicap.

## Quickstart

```bash
pip install -e .[dev]

llamanat cache                # exact KV-cache arithmetic, no training
llamanat demo                 # train one block, print its generated carry chains
llamanat alphabet             # the same recall task in two answer alphabets
python experiments/run_study.py    # the full 3-seed matrix -> results/anatomy.json
python experiments/make_report.py  # print the README block from that JSON
python experiments/make_report.py --write   # and splice it into README.md
pytest -q
```

## How the numbers get here

Everything below the next heading is produced by `experiments/run_study.py`,
committed as [`results/anatomy.json`](results/anatomy.json), and rendered
verbatim by `experiments/make_report.py`. Three CI tests hold that shut: one
byte-compares this block against the renderer's output on that JSON, one asserts
the JSON's settings match the committed study plan, and one asserts the renderer
contains no hardcoded measurement at all — so the prose, including which variant
leads, is derived from the data, and a rerun that flips an ordering flips the
sentence with it.

That is the whole of the reproducibility claim, and it is worth stating exactly
what is *not* inside it. No test re-fits the matrix, so nothing here promises
that your run reproduces these curves. What was checked by hand is that one
process is internally exact: the study re-fits the reference model through four
independent code paths — the matrix, the extrapolation probe, the rotation sweep,
and the answer-alphabet control — and their per-seed numbers agree to the last
decimal, which is what makes the tables one experiment rather than four opinions
about it. `tests/test_readme.py` asserts that agreement on the committed file. A
full rerun at the environment the block below names reproduced all twelve
in-domain cells and all six transfer cells exactly the same way.

The other half of the picture is why the tables print every seed beside the mean.
Float reduction over a batch depends on the build and the thread count, and an
earlier rerun made under different settings moved four of the twelve in-domain
cells — always a seed sitting close enough to an acquisition threshold that the
order of addition decides it, and in one case a single seed of the addition
`layernorm` row going from 0.000 to 0.505. So the README reports the curves and
the seed lists rather than a guarantee about your machine, and the block below
names the python, torch and thread count these fits actually ran under.

## Results

<!-- RESULTS:START -->
*3 seeds (0, 1, 2); one training trajectory per row, scored on held-out examples at checkpoints [300, 600, 900, 1200]; batch 64, lr 0.001, 1200 steps. Produced on cpu by `experiments/run_study.py` under python 3.13.7 / torch 2.14.0+cpu / 4 threads (Windows-11-10.0.26200-SP0), committed as [`results/anatomy.json`](results/anatomy.json), and rendered by `experiments/make_report.py` -- a test fails if this block and that file disagree, and only a rerun *in that environment* is expected to reproduce the numbers bit for bit.*

Reference model: d_model 64, 2 layers, 4 query heads sharing 2 KV heads (head_dim 16), ff 192, rms norm, RoPE -- **102,528 parameters** trained on 4,000 examples.

### What each row replaces

| variant | change against the reference | parameters |
|---|---|---:|
| `llama` | the reference model | 102,528 |
| `abs_pos` | `rope` False (instead of True) | 105,600 |
| `gelu_mlp` | `mlp` gelu (instead of swiglu) | 102,528 |
| `layernorm` | `norm` ln (instead of rms) | 102,848 |
| `mha` | `n_kv_head` 4 (instead of 2) | 110,720 |
| `mqa` | `n_kv_head` 1 (instead of 2) | 98,432 |

The control that makes the `gelu_mlp` row mean anything: it is the same size to the parameter -- 102,528 against SwiGLU's 102,528 -- because a two-matrix gelu network matches three-matrix SwiGLU at `hidden = 1.5 * ff`. That row therefore measures *gating*, not size. `abs_pos` is larger by exactly its position table (3,072 parameters = 48 rows x 64), and `mha` / `mqa` differ only in how many KV heads the projections write.

### In-domain acquisition

**addition** -- 4,000 three-digit sums written as a least-significant-column-first carry chain; graded by the exact sum after free-running generation, no teacher forcing, on 200 unseen pairs.

| variant | params | 300 | 600 | 900 | 1200 |
|---|---:|---:|---:|---:|---:|
| `llama` | 102,528 | 0.000 | 0.138 ± 0.196 | 0.542 ± 0.331 | 0.662 ± 0.411 |
| `abs_pos` | 105,600 | 0.422 ± 0.300 | 0.990 ± 0.011 | 1.000 | 1.000 |
| `gelu_mlp` | 102,528 | 0.000 | 0.060 ± 0.081 | 0.352 ± 0.252 | 0.517 ± 0.367 |
| `layernorm` | 102,848 | 0.000 | 0.010 ± 0.014 | 0.357 ± 0.258 | 0.798 ± 0.208 |
| `mha` | 110,720 | 0.002 ± 0.002 | 0.005 ± 0.004 | 0.050 ± 0.029 | 0.082 ± 0.006 |
| `mqa` | 98,432 | 0.000 | 0.000 | 0.200 ± 0.269 | 0.270 ± 0.324 |

`abs_pos` leads at 1200 steps on 1.000, +0.202 clear of the nearest row. 4 of the 15 pairings rank differently at 300 steps than at 1200 -- at this budget the rows are ranked by *when* the task arrives as much as by whether it can. 4 of 6 rows are not a mean of three similar runs at 1200 steps -- `llama` [0.080, 0.940, 0.965]; `gelu_mlp` [0.000, 0.730, 0.820]; `mqa` [0.000, 0.085, 0.725]; `layernorm` [0.505, 0.925, 0.965]. On a task learned by transition a seed either crosses the threshold or it does not, so those means describe which seeds finished rather than what the variant is capable of, and the spread has to be read with the number. Read this column as acquisition at one fixed length, not as capability: a task whose length never changes lets a position table memorise which slot owns which column, and that is the shortcut the next table takes away.

**recall** -- 4 key/value pairs then a probe key; graded by whether the emitted token is the value that pair actually held, on 200 unseen sequences. The answer is named with a private symbol (`name` mode); the section below measures what the other formulation costs.

| variant | params | 300 | 600 | 900 | 1200 |
|---|---:|---:|---:|---:|---:|
| `llama` | 102,528 | 0.238 ± 0.045 | 0.465 ± 0.312 | 0.647 ± 0.313 | 0.963 ± 0.052 |
| `abs_pos` | 105,600 | 0.237 ± 0.002 | 0.280 ± 0.018 | 0.548 ± 0.320 | 0.990 ± 0.007 |
| `gelu_mlp` | 102,528 | 0.253 ± 0.027 | 0.240 ± 0.029 | 0.240 ± 0.020 | 0.255 ± 0.011 |
| `layernorm` | 102,848 | 0.230 ± 0.051 | 0.430 ± 0.269 | 0.617 ± 0.316 | 0.838 ± 0.229 |
| `mha` | 110,720 | 0.258 ± 0.014 | 0.585 ± 0.241 | 0.998 ± 0.002 | 1.000 |
| `mqa` | 98,432 | 0.223 ± 0.018 | 0.403 ± 0.245 | 0.743 ± 0.332 | 0.743 ± 0.363 |

`mha` leads at 1200 steps on 1.000, +0.010 clear of the nearest row. 5 of the 15 pairings rank differently at 300 steps than at 1200 -- at this budget the rows are ranked by *when* the task arrives as much as by whether it can. 2 of 6 rows are not a mean of three similar runs at 1200 steps -- `mqa` [0.230, 1.000, 1.000]; `layernorm` [0.515, 1.000, 1.000]. On a task learned by transition a seed either crosses the threshold or it does not, so those means describe which seeds finished rather than what the variant is capable of, and the spread has to be read with the number. 1 row never left the floor of this table -- `gelu_mlp` 0.253 at 300 steps, 0.255 at 1200. The first checkpoint and the last print the same number, so that is a flat curve rather than a slow one, and the honest reading is that the variant does not do this task at this budget. That is why the curve is reported rather than a single number: on a task with a threshold, one snapshot records where each row happens to sit relative to its own transition, not what it is capable of.

### Train short, test long

The same fitted weights, scored on sequences longer than the 4 pairs they trained on, in the same answer alphabet as the table above. Nothing was retrained or retuned between columns. The absolute table is sized to 48 positions, so rows outside the trained range exist and were simply never updated -- that is the comparison, not a handicap.

| variant | 4 pairs *(trained)* | 8 pairs | 12 pairs |
|---|---:|---:|---:|
| `llama` | 0.963 ± 0.052 | 0.870 ± 0.088 | 0.715 ± 0.100 |
| `abs_pos` | 0.990 ± 0.007 | 0.260 ± 0.086 | 0.180 ± 0.060 |

At the trained length the two score 0.963 / 0.990, and out of length `llama` holds 0.715 while `abs_pos` falls to 0.180. A rotary block has nothing to memorise about absolute index, so relative distance is all it ever learns; a position table is the cheapest place in the model to hide a positional shortcut, and running past the trained length is where it gets taken away. Note that this is a *transfer* result and it reverses in domain, where `abs_pos` is the easier of the two to fit.

### How fast the rotation runs

RoPE arrives with one number attached to it, the base of the frequency series, and the reference model runs at 10,000. It is the only axis in this repository that costs nothing: no parameter, no bias, no change of shape, just a different set of angles on the same weights. So the sweep runs in both directions from it, on the recall task at the matrix's own budget. The second column is arithmetic rather than measurement: how far two tokens can sit apart before the *slowest* channel pair completes a turn, past which that pair reads two offsets as one angle.

| RoPE base | slowest period (positions) | parameters | 300 | 600 | 900 | 1200 | 8 pairs | 12 pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 47 | 102,528 | 0.233 ± 0.017 | 0.398 ± 0.186 | 0.542 ± 0.271 | 0.683 ± 0.304 | 0.302 ± 0.159 | 0.180 ± 0.084 |
| 100 | 353 | 102,528 | 0.233 ± 0.044 | 0.247 ± 0.029 | 0.627 ± 0.275 | 0.808 ± 0.268 | 0.590 ± 0.287 | 0.373 ± 0.208 |
| 10,000 | 19,869 | 102,528 | 0.238 ± 0.045 | 0.465 ± 0.312 | 0.647 ± 0.313 | 0.963 ± 0.052 | 0.870 ± 0.088 | 0.715 ± 0.100 |
| 1,000,000 | 1,117,326 | 102,528 | 0.237 ± 0.041 | 0.275 ± 0.007 | 0.422 ± 0.226 | 0.978 ± 0.013 | 0.912 ± 0.067 | 0.755 ± 0.102 |

At 4 pairs trained and 1200 steps the rows differ in one float and nothing else: all 4 of them carry exactly 102,528 parameters, because a rotation base adds none. In domain, 1,000,000 prints the best number (0.978 against the paper's 0.963), but the +0.015 gap is inside the 0.052 seed spread beside it and only 1 of 3 paired seeds point that way, so it is not a lead. At 12 pairs, 1,000,000 prints the best number (0.755 against the paper's 0.715), but the +0.040 gap is inside the 0.102 seed spread beside it and only 2 of 3 paired seeds point that way, so it is not a lead. The slowest channel pair turns once every 47 positions at 10 and 1,117,326 at 1,000,000 -- every base here is still inside its first turn at 37 tokens, so what the sweep varies is how much of the window the slowest pair resolves, not whether the sequence wraps. Read seed against seed rather than mean against mean -- the swept fits share their initialisation and their data stream, so two rows at the same seed differ only in the angles -- and only one end of the sweep is decided: 10 lands below the paper's base on all 3 seeds (-0.065, -0.030, -0.745), while the top row here, 1,000,000, is ahead of it on 1 of 3. So this window measures a floor for the rotation base and not a ceiling: running short of range costs real accuracy, and buying more of it than the sequence needs is free. None of this argues for a different base in a real model -- that number is chosen for the context the model has to run, and a window of 37 tokens is not the trade LLaMA was making. What it does show is that an axis with no parameters and no shapes at all still moves the measured accuracy -- the 4 bases spread 0.295 in domain -- which is the cleanest case in this repository for 'the geometry is doing something' rather than 'the extra weights are'.

### How the answer is asked

One seed (0), 4 pairs trained, the same 1200-step budget and the same checkpoints as the tables above. This is a control on the task rather than on the model, and it is the reason the recall rows above are worth reading: the formulation nobody would question -- emit the token you were shown -- is the one a block this size cannot fit, while naming the value with a private symbol reaches 1.000.

| answer alphabet | model | parameters | 300 | 600 | 900 | 1200 | 8 pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| name the value | `llama` | 102,528 | 0.265 | 0.905 | 1.000 | 1.000 | 0.970 |
| name the value | `abs_pos` | 105,600 | 0.235 | 0.305 | 0.310 | 0.985 | 0.180 |
| copy the value token out of the context | `llama` | 102,528 | 0.305 | 0.295 | 0.295 | 0.345 | 0.130 |
| copy the value token out of the context | `abs_pos` | 105,600 | 0.230 | 0.275 | 0.270 | 0.250 | 0.040 |
| copy the value token out of the context | `llama`, tie_embeddings=False | 106,432 | 0.295 | 0.320 | 0.255 | 0.300 | 0.145 |

Same sequences, same architecture, same budget; only the alphabet the answer is written in changes. Naming the value reaches 1.000 in domain and 0.970 at 8 pairs, while copying the value token back out of the context reaches 0.345 and 0.130. The lookup those two share is identical, so the difference is confined to the readout, and untying the shared embedding table moves it the wrong way (-0.045), so the shared table is not the bottleneck and no mechanism for the wall is claimed here.

### Where the KV cache goes

Arithmetic rather than measurement, so it carries no seed noise: 2 layers, batch 1, 2,048 positions, fp16, K and V both stored.

| attention | KV heads | cache per sequence | share of MHA | KV projection parameters |
|---|---:|---:|---:|---:|
| MHA | 4 | 1,024.0 KiB | 1.00 | 16,384 |
| GQA *(reference)* | 2 | 512.0 KiB | 0.50 | 8,192 |
| MQA | 1 | 256.0 KiB | 0.25 | 4,096 |

Half the KV heads costs 8,192 projection parameters -- 8.0% of this model -- and buys 50% less cache; MQA buys 75%. Cache grows with sequence length and KV-head count while the weights stay put, so sharing KV heads becomes worth more the longer the context is -- and it is the one axis here where the parameter budget genuinely moves.

What this table prices is the cache. The accuracy side of the very same axis is in the tables above, and at this budget it does not order itself the same way twice: addition runs `llama` (0.662) > `mqa` (0.270) > `mha` (0.082); recall runs `mha` (1.000) > `llama` (0.963) > `mqa` (0.743). Three seeds at one step budget is not enough to say why, and no reason is offered here.
<!-- RESULTS:END -->

## What this does not show

- **Not a scale claim.** Two layers, four heads and 102,528 parameters trained
  on 4,000 examples. LLaMA's own evidence is a scaling law; nothing here says a
  mechanism that wins or loses at this size behaves the same at 7B. What it does
  say is checkable, because it is one command away.
- **Not a ranking of attention variants.** The KV-head rows are the least stable
  thing in the repository: `mha` is the best recall row and the worst addition
  row, with `llama` between them on one task and above them on the other. Three
  seeds at one budget is a measurement of *this* experiment, so it is published
  with its per-seed spread and without a story.
- **Ordering is budget-dependent.** A row that is behind at 1,200 steps may be
  ahead at 3,000; that is what the checkpoints column is for. The conclusion is
  the shape of the curves, not any single cell.
- **The rotation sweep is a short-window result.** Its longest sequence is 37
  tokens, and a base of `10,000` gives the slowest channel pair a period of
  about 20,000 positions -- LLaMA's number is chosen for contexts dozens of
  times longer than anything measured here, so whatever the sweep ranks first it
  is a statement about a lookup task at a toy window, not a recommendation about
  how to set `rope_theta` in a real model.
- **The task itself is a variable.** The `How the answer is asked` table is the
  one result to read before the others: at this size the model will not fit the
  copy formulation of a lookup it fits perfectly when the answer is named. Any
  claim about a *mechanism* here is therefore conditional on a formulation
  choice that is easy to make unconsciously, and this repository has already
  been burned by exactly that (see the label bug pinned by
  `tests/test_tasks.py`).
- **Reproduction is per-environment.** A rerun is bit-exact at the python, torch
  and thread count the block names, not under an arbitrary one; elsewhere the
  near-threshold seeds can land on either side of their transition. Compare a
  fresh run against the published seed lists rather than against a mean.
- **No tokenizer, no dataset, no pretrained weights.** The vocabulary is 61
  symbols invented for these two tasks; `VOCAB` is the only reason `Config.vocab`
  exists.

## Layout

```
src/llamanat/
  positional.py   RoPE (frequencies, interleaved rotation) and the learned table
  blocks.py       RMSNorm / LayerNorm, causal GQA attention, SwiGLU and GeLU MLPs
  model.py        Config, VARIANTS (the ablation), TinyLLaMA, the init rule
  tasks.py        the two verifiable tasks and their exact graders
  train.py        one loop, checkpoint scoring, free-running generation, probes
  kv.py           the KV-cache arithmetic
  study.py        the measurement plan: seeds, budget, matrix, controls
  cli.py          demo / alphabet / cache / study
experiments/
  run_study.py    writes results/anatomy.json
  make_report.py  renders the README block from it
tests/            graders, blocks, positional maths, training guards, README drift
```

## License

MIT.
