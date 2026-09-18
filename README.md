# typed-decisions

**A lightweight reproduction of the Jev shape — one program state, many typed questions
(`Choice` / `Score` / `Noul`), calibrated probability distributions back in a single parallel
pass — on two backbones, so their speed, accuracy/calibration and training cost can be read side by
side.** Subject-plane name (`kotoba-lang/typed-decisions`): the subject is the typed-decision model,
not a role and not an origin — Jev (typesafe.ai) is the reference point, not a spec we implement
(there is no public spec; the API shape is reconstructed from their site and the dev.to guide, see
`schema.py`). Decisions and the measured record: superproject ADR
`adr-2609181600-typed-decisions-jev-shape-modernbert-vs-llada-moe`.

Nearest repos and the boundary: `kotoba-lang/dllm-qwen38` (block-diffusion adaptation of an AR
model; here the dLLM is used as-is with LoRA, nothing about diffusion training is new),
`kotoba-lang/murakumo` (serves GGUFs; nothing here serves), `cloud-itonami/llm-dataset` (large
checkpoints; nothing large is committed here — the corpus is rebuilt from public datasets by
`data.py`, runs live in the Modal volume `typed-decisions-cache`).

## The shape

```
state  +  { q1: Choice(instr, [opt...]), q2: Score(instr, [level...]), q3: Noul(instr) }
  -> { q1: {choice, probabilities, confidence}, q2: {score, probabilities, confidence}, q3: {noul} }
```

Every kind is one softmax over an option list; `score` reads the distribution out as an expected
level (may land between levels, like Jev's `1.035`), `noul` as p(yes). One head serves all three,
and N questions on a state are answered in one forward.

| | backbone A — encoder | backbone B — dLLM |
|---|---|---|
| model | `answerdotai/ModernBERT-base` (149M) / `-large` (395M), 8k ctx | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` (7.4B total, ~1B active) |
| input | `[CLS][STATE] s [Q] instr [OPT] o1 [OPT] o2 … [Q] … [SEP]` | `<state>…</state>` + `Qi. instr Options: A) … B) …` + `Ai: [MASK]` per question |
| read-out | matching head over the **mean of each option's text tokens** × the question's text tokens, softmax within the question (`encoder.py`) | logits at each `[MASK]` slot restricted to the option-label tokens, softmax (`dllm.py`) |
| passes | 1 | 1 (steps=1); steps>1 = LLaDA low-confidence remasking, measured too |
| trained | full fine-tune, fp32 + bf16 autocast | LoRA r16 on q/k/v/o/gate/up/down, bf16 |
| loss | CE + Brier (same for both) | CE + Brier (same) |
| structured-output errors | 0 by construction | 0 by construction |

Post-hoc temperature is fitted on the validation split and reported as its own row (`Tfit`).

## What the labels are (no synthetic answers, no teacher)

| source | state | questions (gold from the dataset's own label) |
|---|---|---|
| `mteb/banking77` (PolyAI data) | customer message | Choice intent (77 options) · Choice product area (10, deterministic keyword rule over the 77 intent names) · Noul "asks about a card" |
| `SetFit/sst5` | review sentence | Score sentiment level (5 ordered) · Choice polarity (3) · Noul "expresses a positive opinion" |
| `google/boolq` | passage | Noul = the dataset's question |

6,000 train / 400 val / 1,000 test states per source → 18,000 / 1,200 / 3,000 states, 42,000 / 2,800 /
7,000 questions (`data.py`, seed 0, test shuffled across sources). Jev's own number (67.8% on four
private workflows, agreement with frontier models) is not reproducible from outside; these are
public gold labels, so the accuracies below are not comparable to that column.

## Run

```
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest -q tests                 # tiny random models, CPU, ~4 min, no download
.venv/bin/python -m typed_decisions.data --out data # the corpus (three HF datasets, seconds)
modal run modal_app.py::data                        # same, into the Modal volume
modal run modal_app.py::encoder --model answerdotai/ModernBERT-base --lr 5e-5 --epochs 2
modal run modal_app.py::dllm --limit 6000 --batch 8 --grad-accum 1 --eval-steps 1,2
```

Every run writes `report.json` (args, data counts, loss curve, train wall/tokens/s/peak-mem/USD,
train-subset and test metrics per kind and per source, temperature, latency rows, throughput rows);
the local copies of the runs cited below are in `reports/`. USD = H100 wall seconds ×
$0.001097 (modal.com/pricing, read 2026-09-18) — nothing else is in that number.

## 要約（日本語）

**何を作ったか。** Jev（typesafe.ai）の形 —— 1 つの program state に N 個の型付き question
（`Choice` 最大 255 択 / `Score` 2〜10 順序段階 / `Noul` yes-no）を載せ、**1 forward** で
question ごとの較正済み確率分布を返す model —— を、同じ loss（CE + Brier）・同じ読み出し・
同じ latency bench で 2 系統の backbone に載せて実測した。encoder（ModernBERT-base/large、
対照に DeBERTa-v3、RoBERTa）は full fine-tune、dLLM（LLaDA-MoE-7B-A1B）は LoRA r16 で
question ごとに `[MASK]` 1 slot を置く。label は公開 dataset の gold のみ（banking77 / sst5 /
boolq、train 18,000 state / 42,000 question）。生成しないので structured-output error は
どちらも構造的に 0。

**結果（H100、Modal、test 1,500 state / 3,508 question）。**

| | ModernBERT-base 149M · 2 ep | DeBERTa-v3-large 435M · 1 ep | LLaDA-MoE-7B-A1B LoRA · 1 ep · 6k |
|---|---|---|---|
| 精度（全 question） | 0.717 | **0.855** | 0.835 |
| Brier / ECE（T fit 後） | 0.359 / 0.013 | **0.204 / 0.014** | 0.231 / 0.028 |
| e2e latency、1 state × 10 q、p50 | 68 ms | **42 ms** | 676 ms |
| forward のみ、10 q / 100 q | **19 / 34 ms** | 39 / —（512 ctx に入らない） | 846 / 841 ms |
| throughput、batch 8 | 435 q/s | 460 q/s | 28 q/s |
| 訓練費（H100 $0.001097/s） | **$0.16** | $0.26 | $2.29 |
| 訓練 question 1k あたり | $0.002 | $0.006 | $0.163 |

等データ対照（6k state × 1 ep）: 0.577 / 0.819 / 0.835。LLaDA の diffusion steps=2 は +0.3 pt で
latency 2 倍（1,341 ms）—— 1 pass が動作点で、これは Jev 自身の主張と同じ。LLaDA の zero-shot は
0.645（boolq だけ 0.829）。

**前提を覆した測定。** 「ModernBERT-large が本命」は成り立たなかった。新設 `[OPT]` marker token の
hidden state を採点する head（最初の設計）は、lr 1e-5〜1e-4 / head lr / Brier 重み 0・1・3 /
autocast 有無 / sdpa・eager / `reference_compile` 有無 / 1〜2 epoch の全掃引で label prior から
動かない（banking77 intent ≈ 0.12、boolq ≈ 0.62 = 多数派）。loop 自体は正しい（16 state を 50 step で
loss 0.000 に過学習、train-subset acc 1.0）。効いたのは読み出しの変更 —— **option の text token の
平均**（と question text の平均）を読む `--pool span`（現在の既定）。この head で ModernBERT-base は
即座に学び（3k/1ep で 0.539）、DeBERTa-v3-large は 0.787、RoBERTa-large は 0.710 —— ModernBERT-large
だけが 0.39 のまま。ModernBERT-base を 6 epoch 回すと 0.746 だが、短文 2 source を暗記して boolq は
0.63 で平ら、ECE は 0.12 に悪化（train-subset 0.949）。

**結論。** 製品経路は encoder。ただし **DeBERTa-v3-large**（最良: 0.855 / 42 ms / $0.26）であって
ModernBERT-large ではない。ModernBERT-base は最安・8k context の選択肢（0.717 / 68 ms / $0.16）。
dLLM は精度で並ぶが latency 16 倍・訓練費 14 倍。Jev の 70〜500 ms band には encoder なら桁で
余裕があり（e2e は Python の tokenise が支配、model は 19〜39 ms）、7B dLLM は 1 step でも band の
外。Jev の $0.000081/task は ModernBERT-base の 100 決定 forward（34 ms = $0.00004/pass、
1 決定 $0.0000004）の約 200 倍 —— 価格であって原価ではない。ModernBERT-like 1〜2B を pretrain する
案は、この結果の前では根拠が無い。

**測っていないもの。** OOD question（同じ state に未見の instructions / option 列 —— model が
question を読んでいるか slot を暗記したかを分ける検査）、frontier teacher の蒸留（Jev の
67.8% はこの target）、複数 seed、ModernBERT-large の多 epoch、Apple M1 Max（MPS）の latency
（2 回とも session 再起動で死んだ）。

## Measured (H100 80GB on Modal, 2026-09-18; `reports/*.json`)

Test = 1,500 states (500 per source; `--test-limit 1500`), 3,508 questions. "e2e" latency is
from Python strings (tokenise + collate + forward), "forward" is the model alone on a pre-collated
batch; batch 1, one state, N questions packed on it, p50 / p95 of 20 repeats after 3 warm-ups.

### The three backbones, head to head

| | ModernBERT-base 149M · 18k states · 2 ep | DeBERTa-v3-large 435M · 18k · 1 ep | LLaDA-MoE-7B-A1B (LoRA r16) · 6k · 1 ep |
|---|---|---|---|
| accuracy, all 3,508 q | **0.717** | **0.855** | **0.835** |
| Brier / ECE (T fitted) | 0.359 / 0.013 | 0.204 / 0.014 | 0.231 / 0.028 |
| banking77 intent (77 options) | 0.833 | 0.922 | 0.890 |
| banking77 "about a card" noul | 0.942 | 0.968 | 0.952 |
| sst5 level, acc / MAE (levels) | 0.375 / 0.85 | 0.585 / 0.51 | 0.575 / 0.54 |
| sst5 polarity | 0.628 | 0.800 | 0.783 |
| boolq noul | 0.649 | 0.881 | 0.869 |
| zero-shot (untrained), all | — | — | 0.645 (boolq 0.829) |
| **latency, 1 state × 10 q, e2e p50 / p95** | **68 / 74 ms** | **42 / 47 ms** | **676 / 706 ms** |
| latency, 1 × 100 q, e2e | 95 / 103 ms | does not fit 512 ctx | 709 / 755 ms |
| forward only, 1 × 10 q | 19 ms | 39 ms | 846 ms |
| decisions / s at 1 × 100 q (forward) | 2,931 | — | 119 |
| throughput, batch 8 states (1–3 q each) | 435 q/s | 460 q/s | 28 q/s |
| throughput, batch 32 | 1,180 q/s | 583 q/s | — |
| train wall / seq-tok/s / peak GiB | 150 s / 53k / 7.7 | 237 s / 16k / 33 | 2,083 s / 0.94k / 53 |
| **train cost (H100 $)** | **$0.16** for 84k question-passes | **$0.26** for 42k | **$2.29** for 14k |
| $ per 1k train questions | $0.002 | $0.006 | $0.163 |
| context | 8,192 | 512 (state cut to 256 tok) | 4k+ |

Equal-data control (6,000 train states, 1 epoch, all three): ModernBERT-base 0.577 ·
DeBERTa-v3-large 0.819 · LLaDA-MoE 0.835. The dLLM's edge over the best encoder at equal data
is 1.6 points; its edge in latency is −16×.

Diffusion steps on LLaDA (10 questions): steps=1 0.835 · steps=2 0.838 · steps=4 (1,500 states)
0.633→ — the smoke run showed steps=4 *lower* (0.633 vs 0.650 at 60 q); the full run's steps=2 gains
0.3 points for 2× latency (1,341 ms). One pass is the right operating point, which is what Jev claims
for itself.

### The ablation that mattered: what the head reads

ModernBERT with a scoring head on a fresh `[OPT]` marker token — the design the prompt sketched —
**does not learn** in one epoch of 3,000 states, at any learning rate (1e-5…1e-4), head lr (3e-5,
1e-3), Brier weight (0, 1, 3), autocast on/off, sdpa/eager, `reference_compile` on/off, on base
and on large: every run ends at the label prior (ce ≈ 1.6, banking77 intent ≈ 0.12, boolq ≈ 0.62 =
the majority class). The same loop overfits 16 states to loss 0.000 in 50 steps, so the mechanics
are right; the marker's hidden state simply has no pretrained structure to score. Reading the
**mean of each option's text tokens** (and of the question's text) instead — `--pool span`,
now the default — makes ModernBERT-base learn immediately (3k states / 1 ep: 0.539, banking77
0.550). ModernBERT-**large** still does not move under `span` in 1 epoch of 3k (0.39) and needed
18k × 2 epochs to reach 0.451 with the marker head; DeBERTa-v3-large under the same `span` head
reaches 0.787 on 3k / 1 ep and RoBERTa-large 0.710. The MLM-slot route (ModernBERT-base through
the dLLM code path, answer token at a `[MASK]`) is at the prior too (0.392).

| 3k states, 1 epoch, `span` head | acc all | b77 intent | sst5 level | boolq |
|---|---|---|---|---|
| ModernBERT-base, lr 5e-5 | 0.539 | 0.550 | 0.193 | 0.622 |
| ModernBERT-large, lr 1e-5 / 3e-5 / 5e-5 / 1e-4 / 2ep / no-amp / eager | 0.388–0.399 | 0.074–0.131 | 0.21–0.24 | 0.617–0.628 |
| DeBERTa-v3-base, lr 3e-5 | 0.398 | 0.144 | 0.208 | 0.622 |
| DeBERTa-v3-large, lr 2e-5 | **0.787** | 0.809 | 0.530 | 0.842 |
| RoBERTa-large, lr 2e-5 | 0.710 | 0.720 | 0.381 | 0.699 |

So the "ModernBERT-large is the obvious first pick" premise did not survive measurement:
on this pipeline it is the slowest learner of the five encoders tried, and the 149M base is both
faster and better per training dollar. More epochs on base (`enc-base-18k-6ep`, 6 ep, $0.43):
0.746 all / banking77 0.894, but boolq stays at 0.631 and sst5 level at 0.429 while the
train-subset accuracy is 0.949 and ECE rises to 0.12 — it memorises the two short-text sources
and does not read the boolq passages. DeBERTa-v3-large at 1 epoch (0.855, boolq 0.881) is the
better use of the same $0.26–0.43. ModernBERT-large with more epochs is not measured.

### Reading against Jev's public numbers

Jev's page: 70–500 ms end-to-end, $0.000081 / task, 0 structured-output errors, 67.8% on their own
four workflows (agreement with frontier models — not comparable to gold-label accuracy here).
Both encoders sit inside that latency band on an H100 with room to spare (ModernBERT-base forward
19 ms for 10 decisions; e2e is dominated by Python tokenisation, not the model); the 7B dLLM does
not (0.6–0.9 s at steps=1). Structured-output errors are 0 for all three by construction. At
H100 $0.001097/s, ModernBERT-base's 100-decision forward (34 ms) is $0.00004 per pass, i.e.
~$0.0000004 per decision — the per-task number Jev quotes is 200× above what the compute costs,
which is consistent with it being a price, not a cost.

### What this does not show

- No frontier-teacher distillation: labels are dataset gold. Jev's "agreement with GPT/Claude"
  column is a different target and needs a teacher run (Qwen3.8-27B via murakumo would be the
  workspace route) — not done.
- No OOD split: every test question type was seen in training. A held-out *question* (new
  instructions, new option set on the same states) is the test that tells whether the model reads
  the question or memorised the slot; not done.
- One seed, one H100 node; run-to-run throughput variance on Modal was ±20% in
  `dllm-qwen38`'s measurements and is not re-measured here.
- LLaDA was LoRA r16, one epoch on a third of the data; the encoders were full fine-tunes on all
  of it. The equal-data control above is the fair row; the head-to-head is the "what you'd
  actually run" row.
- Local Apple M1 Max (MPS) latency for ModernBERT-base was attempted twice and both runs were
  killed by session restarts before the bench row was written; not measured. The H100 numbers are
  the ones in the tables.
