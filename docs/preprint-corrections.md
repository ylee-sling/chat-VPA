# Corrections for the next revision of the Chat-VPA preprint

For the draft of 22 November 2026 (`preprint_261122_chat_vpa.pdf`). **The preprint is not edited here**; this is
the list to raise when the author revises it. Until then the code and `docs/e0.md` follow the corrected facts and
do not repeat the draft's wording. Each item says where the number comes from.

## 1. "Two tokens per pass, left to right, cost nothing" was not measured

*Where:* the abstract ("keeps 17.1% when two tokens per pass are decided left to right"), Table 1 (column
"K = 128 (two tokens)"), Finding 1, Section 3.1 ("two tokens per pass next to known context cost nothing"),
Section 3.5 ("at two tokens per pass").

*What is correct.* K = 128 passes over a canvas of 256 positions average two positions per pass, but the cosine
schedule does not spread them evenly. The left-to-right order reveals **one token per pass for the first 42
passes** and two or more only afterwards (*computed* with the sampler's own `reveal_counts`, N = 256; text-VPA
`docs/stage1.md` at `6f97346` states the same). A program of about 48 tokens is therefore written at one token per
pass almost to its end, and the 17.1% at K = 128 left to right repeats the one-token-per-pass measurement. The
predecessor has **no** measurement of two adjacent tokens decided per pass.

*What stands.* The comparison of orders at K = 128 (17.1% left to right, 9.8% confidence, 0.7% exact kernel;
*measured*, `runs/stage0-a`) is a comparison at an equal number of passes, and Finding 1's statement that the
place of a decision matters is supported by it. What does not stand is the reading that two neighbouring tokens
per pass are free: the first measured point with several adjacent tokens per pass is K = 32 (1.6%).

*Consequence for E0.* None for its rules. The regime between one and four adjacent tokens per pass is unmeasured,
which is one more reason for the optional K = B budget of `docs/e0.md`, C2.

## 2. The latent oracle has finished

*Where:* Section 1.2 (Finding 3, "was still training when this draft was written"), Section 3.4 ("at its interim
checkpoint"), Section 7.3, Appendix C (Table 9 and "Latent oracle, interim").

*Final numbers* (*measured*, text-VPA `runs/stage1-latent`, 5,000 steps from `runs/stage0-a/last.pt`, MBPP
validation 90 x 5; text-VPA `docs/stage1.md` at `6f97346`):

| Pass@1 (parses) | K = 8 | K = 32 | K = 128 | one token per pass |
|---|---|---|---|---|
| null plan, same weights | 0.2% (6.9%) | 0.2% (10.7%) | 0.4% (12.7%) | 14.2% (96.7%) |
| latent oracle | 1.8% (17.6%) | 2.2% (22.9%) | 2.2% (26.4%) | 6.4% (51.1%) |
| oracle minus null, points, 95% paired bootstrap | +1.6 [+0.0, +3.3] | +2.0 [+0.7, +3.8] | +1.8 [+0.0, +4.0] | -7.8 [-14.7, -1.1] |

Held-out loss with every position masked, at step 5,000: 2.194 nats with the plan, 4.177 with the null plan,
15.109 with another function's plan. No sample reproduces its reference (0.0% at every K), so by the reading rule
set before that run its gate is **not read**; the ratio at K = 8 is 0.075 [0.018, 0.144].

*What changes in the text.* "0.0% Pass@1 at K = 8" for the latent oracle becomes 1.8% against 0.2% for the null
plan; the draft's sentence that a plan which is read does not help parallel decisions needs the qualifier that the
latent plan helped by about two points at K <= 128 and cost 7.8 points at one token per pass. This oracle
conditions on the reference and stays a diagnostic.

## 3. The whole-canvas reveal counts at K = 32 (a discrepancy in the predecessor's notes, immaterial)

text-VPA `docs/stage1.md` lists the left-to-right counts at K = 32 as 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7 and 7
"over the first 59 positions". The ported sampler's rule gives **8** for the fourteenth pass and 60 positions
(*computed*: 204 masked positions times a jump mass of 0.0376 is 7.67, which rounds to 8). "Four to six adjacent
tokens per pass" remains a fair description of passes 7 to 11, which cover positions 13 to 38; a program of 48
tokens ends in pass 13, at seven per pass. If the preprint quotes the counts, it should quote the computed ones.

## 4. Table 5 lacks the rows that H2 needs

Section 6.2 evaluates each arm at four (B, K) cells, and H2 compares the arms at equal (B, K) for g = 4, 8 and 16.
Table 5 has the human-reference arm at (16, 4) only. `docs/e0.md`, Section 6, has the full grid; the preprint's
table should gain the three rows, or H2 should say that it is read from a table in the appendix.

## 5. Table 5's ratio 0.694 has another denominator than its column

The column "Ratio to AR" has 1 in the row of the released `generate` (23.8%), and its 0.019 for the K = 8 row is
that row over `generate` (*measured*, text-VPA `docs/stage0.md`: "H2's pair"). The 0.694 [0.450, 0.965] of the
left-to-right row is the predecessor's gate rho_1, defined there as 17.1% over the **code check** (24.7%). Against
`generate` the ratio is about 0.72 from the rounded percentages (*computed*); its exact value and interval are to
be recomputed from the sweep's samples with `chat_vpa/report.py`. Finding 1's sentence in Section 1.2 names both
baselines before "so the ratio is 0.694" and should name the code check. H1's threshold is against `generate`, and
`docs/e0.md` uses that denominator throughout.

## 6. Points where the draft is silent and E0 needs a decision (not errors)

Listed in `docs/e0.md`, Section 3, for the author: which budgets E0 trains on (Table 2 has {8, 6} for a stage that
is later halved; E0 has no halving); whether one set of adapters serves both block sizes; R in E0; how "grows" in
H2 and "near the floor" in the reading of a failed H1 are decided; how many slots the prefix-growing and
confidence orders reveal per pass. One fact worth a sentence in Section 3.1: with the cosine schedule the passes
are unequal, so at (16, 4) the last pass decides 6 of the 16 slots and at (32, 2) the last pass decides 22 of 32
(*computed*), while the bound (10) is stated in the largest number per pass, not in the mean g = B/K.
