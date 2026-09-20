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

## 第 2 反復（2026-09-18）: OOD question・教師・code decision —— ADR-2609181800

前回の「測っていないもの」のうち 3 つを測った。全部 `reports/ood-*.json` / `reports/code-*.json` /
`data/teacher-test.jsonl`。

### ① OOD question（`data/ood-test.jsonl`、`data.py ood_questions`）

同じ test state に**未見の instructions と未見の option 列**（banking77: 4 topic + 「金が出ていく
話か」noul + 「銀行の行動が要るか」3 段 score / sst5: 「友人に薦めるか」noul・tone 3 択・強度 3 段 /
boolq: **否定形** noul「この主張は passage によれば偽か」+ supported/contradicted）。gold は同じ
dataset label からの決定論規則。

| test 1,500 state | in-domain | **OOD** | b77 topic（多数派 0.33） | b77 outflow（0.67） | boolq 否定 noul（0.64） | boolq support（0.64） | sst5 tone（0.41） | sst5 推薦 noul（0.59） | score 2 種 |
|---|---|---|---|---|---|---|---|---|---|
| ModernBERT-base 2 ep | 0.728 | **0.485** | 0.400 | 0.418 | **0.397** | 0.579 | 0.540 | 0.690 | 0.378 / 0.474 |
| DeBERTa-v3-large 1 ep | 0.852 | **0.622** | 0.578 | 0.685 | 0.671 | 0.750 | 0.757 | 0.905 | 0.269 / 0.360 |
| LLaDA-MoE-7B-A1B LoRA | 0.837 | **0.614** | 0.669 | 0.827 | **0.391** | 0.720 | 0.741 | 0.875 | 0.285 / 0.397 |

読み方: ModernBERT-base は OOD でほぼ多数派以下 —— slot を暗記している。**否定形 noul は
base も dLLM も多数派を割る（0.40 / 0.39）**: 否定を読まず元の question に答えている。DeBERTa だけ
0.671。新しい level 列の **score は 3 model とも多数派以下** —— 期待段階の読み出しは option の
順序を学んでおらず、未見の段階列に転移しない（`score` は OOD では壊れている）。sst5 の「薦めるか」
だけは全 model で転移する（意味が「positive か」と同じ）。

### ② 教師 `qwen3.8-flash-next-whitehacker`（`teacher.py`）

lane の実測（2026-09-18）: `logprobs` **無し**、`n` は 1 固定、temperature 1.0 で 8 sample が**全部同一**
→ 教師から取れるのは **hard label だけ**（分布は取れない）。urllib 既定 UA は Cloudflare 1010 で
403、client 名を名乗る UA で通る。throughput は並列 8 で **0.16 req/s**（p50 47 s / p95 71 s、
lane が直列化している）、途中 503 あり。答えの形式ゆれ（`A1:` を全行に付ける、`Q1.`、散文）で
最初の parser は 19% を落とし、行指向 parser で 4% まで下げた。

| 教師の gold 一致（test、hard label） | n | acc | 参考: student DeBERTa-v3-large |
|---|---|---|---|
| banking77 intent（77 択） | 590 | **0.776** | 0.922 |
| banking77 card noul | 293 | 0.877 | 0.968 |
| sst5 level（5 段） | 119 | 0.555 | 0.585 |
| sst5 polarity | 115 | 0.783 | 0.800 |
| sst5 positive noul | 115 | 0.861 | 0.903 |
| boolq noul | 29（503 で途切れ） | 0.828 | 0.881 |

**教師は zero-shot では in-domain gold で student に負けている**（77 択 intent で 15 pt 下）。
この教師から hard label を蒸留すると in-domain 精度は**下がる**。教師の価値は gold の無い
question（= OOD・新規 workflow）のラベル付けにしか無く、そこでの教師の精度は未測定
（OOD split は規則 gold なので測れる —— 次の反復）。コストは GPU ではなく壁時計: 18k state で
**約 31 時間**（0.16 req/s）、10⁵ state なら約 1 週間、金額は無料枠 + owner coupon で $0。

### ③ code decision（`code_data.py`、`data-code/`）

symbol-index（v5、618k symbol、`.kotoba-cache/symbol-index.tsv`）から「**definition は次にどの
definition を参照するか**」を Choice にした。state = `ns/name` + docstring（あれば）+ 既知の参照
（1 つ hold-out）、option = hold-out 参照 + 同じ namespace の兄弟 def が参照する定義から
distractor（型情報は index に無いので「同 ns 近傍」で代用）、noul = 「X を参照するか」yes/no を
**別 example に分離**（同居させると yes-noul の X が Choice の答えを漏らし train loss 0.000 になった —
実測）。split は **namespace 単位の hash**（test ns は訓練に出ない）。413k def → 参照 2 本以上
68k Choice + 68k noul pair、test 1,407 ns。

| train 30k example・1 ep | Choice（k≈5.7、chance 0.18） | noul | ECE | train-subset | 費用 |
|---|---|---|---|---|---|
| DeBERTa-v3-large | **0.638** | 0.809 | 0.019 | 0.822 | $0.19 |
| ModernBERT-base | 0.286（発散、loss 1.5→6.1） | 0.580 | 0.178 | 0.475 | $0.14 |

未見 namespace で 0.638 は「名前と近傍だけ」から出ている数字。型で候補を刈れば option 集合が
縮む（k が下がる）ので上がる余地はそちらにある。ModernBERT-base はここでも発散した（標準 corpus
の 18k × 1 ep 再走でも 0.434 に落ちた run がある = `replicate-std-base-1ep`。**run 間の不安定**）。

### 判断 → code / tool call（`wire.py`）

答えは pointer: `{:proposal/kind :wire-reference :reference {:fq … :hash …} :probabilities {…}
:confidence … :admit? {:noul … :threshold … :decision :autonomous|:escalate} :memo-key …}`。
`memo-key` = sha256(state, question, option hash 列) —— symbol-index の closure hash と同じ流儀で、
同じ入力の判断は forward を走らせない。実行はしない（agent は propose まで）。

## 第 3 反復（2026-09-18）: 言い換え・否定・option 順の augmentation、consistency loss、教師の OOD 精度

`augment.py`（gold を構造的に保つ変換: choice の option 順 shuffle / 言い換え template / distractor の
drop / score 段階名の同義置換 / noul の否定 = gold 反転）を訓練時に確率 p で掛け、`--consistency` で
同じ判断の 2 表層の対称 KL を足す。DeBERTa-v3-large、18k state、test 1,500 state。

| | in-domain | **OOD** | OOD ECE | boolq 否定 noul | b77 topic | b77 outflow | boolq support | 費用 |
|---|---|---|---|---|---|---|---|---|
| baseline（1 ep） | 0.852 | 0.622 | 0.116 | 0.671 | 0.578 | 0.685 | 0.750 | $0.26 |
| **augment p=0.7（1 ep）** | 0.846 | **0.648** | **0.088** | **0.83** | 0.55 | **0.80** | 0.71 | $0.26 |
| augment 0.7 + consistency 0.5（1 ep） | 0.852 | 0.635 | 0.110 | 0.83 | 0.63 | 0.73 | 0.60 | $0.50 |
| augment 0.5 + consistency 0.2（2 ep） | 0.855 | 0.621 | 0.130 | 0.84 | 0.52 | 0.75 | 0.60 | $0.99 |

- **augmentation だけで OOD +2.6 pt、否定形 noul +16 pt、OOD ECE −0.03**、in-domain は −0.6 pt。
  否定は「訓練で見せれば読む」。
- **consistency KL は足しても効かない**（この規模では）。forward 2 倍で費用 2 倍、OOD は同等か下。
  2 epoch も OOD を上げない（in-domain だけ上がる = 暗記側に振れる）。
- **score は依然 OOD で多数派以下**（0.22〜0.38）。ただし教師も同じ 2 問で 0.31 / 0.39 —— 3 model +
  教師が揃って落ちるので、**OOD の score 2 問（b77 urgency / sst5 intensity）の規則 gold 自体が
  怪しい**。この 2 問は次の反復で gold を作り直すまで数字を読まない。

### 教師の OOD 精度（`data/teacher-ood-test.jsonl`、280 state、726 question）

| | 教師 whitehacker | 最良 student（augment） |
|---|---|---|
| OOD 全体 | **0.696** | 0.648 |
| b77 topic | **0.742** | 0.55 |
| b77 outflow noul | 0.859 | 0.80 |
| boolq support | **0.909** | 0.71 |
| boolq 否定 noul | 0.854 | 0.83 |
| sst5 tone | 0.674 | 0.77 |
| sst5 推薦 noul | 0.833 | 0.91 |

教師は **OOD では student より 5 pt 上**（in-domain では 15 pt 下）。差が大きいのは boolq の
passage 読解（support 0.91 vs 0.71）と b77 topic。つまり蒸留の使い道は前反復の結論どおり
「gold の無い question」で、そこでの取り分は **question 種による**: 読解系は教師、感情系は student。
教師の OOD ラベル 300 state は 0.16 req/s で 31 分（p50 11 s、前回の 47 s より lane が空いていた）。

### 判断 → 承認キュー（`wire.py to_kaizen_issue`）

proposal を cloud-itonami の kaizen ingress の形（`{:kind :id :title :body :severity}`、id = memo-key
の先頭 16 桁なので同じ判断は `200 already-open`）に落とす関数を足した。**POST はしていない**
（narrow key では取り消せず、人が cockpit で閉じるまで残るので、実弾は governor 側の受け口を
決めてから）。

### 型で刈った候補列（未着手、理由）

symbol-index に `.kotoba` の定義は 95 行しか無く（618k symbol 中）、kotoba-sema の型で候補を刈れる
corpus が無い。code decision は当面「同 ns 近傍」の distractor で測る。

## 第 4 反復（2026-09-18）: OOD score の gold 作り直し、seed 3 本、承認キューへの実 POST、教師ラベルの追加

### 1. OOD score の gold（`data.py`）
b77 の「urgency」（3 model + 教師が多数派以下）は **削除**（banking77 に順序の ground truth は無い）。sst5 の
「intensity」（|level−2|、同じく全員が落ちる）も削除し、**dataset の 5 段階の単調な relabel** 2 問に置換:
`stars`（1〜5 stars、同方向）と `disappointed`（not at all〜very、**逆方向**）。gold は defensible、
未見の option 文言 + 未見の instructions は保ったまま。多数派 0.255。

### 2. seed 3 本（DeBERTa-v3-large、augment p=0.7、1 ep、新 OOD set）

| seed | in-domain | OOD | OOD ECE | boolq support | boolq 否定 | sst5 score acc / MAE（段階） |
|---|---|---|---|---|---|---|
| 0 | 0.846 | 0.685 | 0.049 | 0.84 | 0.84 | 0.39 / 0.72 |
| 1 | 0.842 | 0.663 | 0.053 | 0.68 | 0.86 | 0.37 / 0.78 |
| 2 | 0.852 | 0.687 | 0.028 | 0.79 | 0.85 | 0.42 / 0.67 |
| **mean（spread）** | **0.847（0.011）** | **0.678（0.025）** | | | | |

in-domain の run 間差は 1 pt、OOD は 2.5 pt —— OOD で 2 pt 以下の差は seed noise の中。boolq support が
0.68〜0.84 と最も揺れる。新しい score 2 問は多数派 0.255 に対して 0.37〜0.42、MAE 0.7 段階 ——
「未見の段階列で方向を読む」は部分的に成立（前の規則 gold では測れていなかった）。

### 3. 承認キューへの実 POST（`wire.py to_kaizen_issue`）
- code decision model（DeBERTa、held-out ns、choice 0.629）から 20 proposal を emit（13/20 正解、
  `data/proposals-code.json`）。1 件目（正解、confidence 0.74）を kaizen ingress の形にした。
- ingress の id 形式は `^kaizen:[A-Za-z0-9:._/-]{1,300}$`（`cloud_itonami.kaizen/validate`）—— id を
  `kaizen:typed-decisions:<memo16>:<window>` に直した。
- **POST は `503 kaizen intake is not provisioned for this tenant`**（`reports/kaizen-post-receipt-20260918.txt`）。
  tenant `kotoba-lang/typed-decisions` の ingress key digest が worker の KV
  （`ITONAMI_DATA` / `kaizen:kotoba-lang/typed-decisions:ingress-key-sha256`）に無い。provisioning は
  「random key を Keychain に置き、その sha256 を `wrangler kv key put` する」の 2 手で、**Keychain
  への書き込みがこの session の権限で拒否された**ので未完。手順は下記「owner がやること」。
- augmentation を code corpus に掛けると **学習しない**（loss 1.5 → 1.7、choice 0.18 = chance、
  `reports/code-deb-large-1ep-aug05-props-*.json`）。code の option は識別子なので言い換え template と
  shuffle が state と option の対応を壊す —— augmentation は自然文 corpus 用。

### 4. 教師ラベルの追加（進行中）
train state 700 件（banking77 350 + boolq 350）に読解系 OOD question（topic / outflow / support /
否定）を生成し（`data/ood-train.jsonl`、1,400 q）、教師でラベル付けして student に足す。lane が
**同じ prompt で 180〜600 s hang**（"Say OK" は 1.2 s）—— 並列 8 で timeout した 32 request が lane の
queue に残って直列化していると読める。probe が 60 s 未満で返るまで待って concurrency 2 で再開する
script を置いた。結果はこの節に追記する。

### owner がやること（provisioning、1 回）
```
KEY=$(openssl rand -hex 32)
security add-generic-password -a cloud-itonami -s "cloud-itonami KAIZEN_INGRESS_KEY kotoba-lang/typed-decisions" -w "$KEY" -U
printf '%s' "$KEY" | shasum -a 256 | cut -d' ' -f1 | xargs wrangler kv key put --namespace-id e9857fe8617440e59f9720293dc53afd "kaizen:kotoba-lang/typed-decisions:ingress-key-sha256"
```
その後 `curl -X POST … -d @issue.json https://itonami.cloud/api/kotoba-lang/typed-decisions/kaizen` で
`202 proposed` を確かめる（issue.json は `data/proposals-code.json[0]` を `to_kaizen_issue` に通したもの）。

## 段 0 / 段 1（2026-09-18）: `.cljc` の eval task と Hermes transcript の判断化 —— ADR-2609181900

### 段 0: `:kotoba` task class（`mine_kotoba_tasks.py` → `scripts/model-eval/kotoba-tasks.edn`）
agent-task-models の eval には 2026-09-18 まで **kotoba / cljc の task が 0** だった（generic な `:code` 11 問）。
workspace の git 履歴から fix pair を採る: 1 commit が `src/*.cljc|cljk` を 1 本だけ触り、その module の test ns が
在り、**repo をその commit で展開して module を差し替えた harness が before=FAIL / after=PASS の両方を返す**
pair だけを task にする（片方しか出ない pair は捨てる —— 壊れた code で通る task は何も測らない）。
80 repo × 400 commit を歩いて **24 task / 12 repo**（amu, kotoba-sema, kotoba-native, aiueos, kototama, inga, engi,
kuro, kotobase-peer, kotobase-server, app-kotoba-cloud, slides）。funnel: fix 系 commit の 65% は src を複数触る、
残りの 3/4 は test ns が無い。

harness（`scripts/agent-task-models-tick.cljk` / `bench.cljk` の `:cljk`）: `git archive <sha>` → module 差し替え →
runner が `clojure.test` の summary 行を数える（sci では `t/report` の再定義が効かない、実測）→ 最終行 `PASS`。
`--self-test-cljk N` が両方向を確かめる（4/4 flip、exit 1 で落ちる gate）。

**最初の測定（8 task、最小のもの、実 tick 経由 = ledger 追記）**:

| model | 結果 |
|---|---|
| nex-n2.5-mini-uncensored | **8/8 truncated**（out=2048 で content 空、finish_reason length —— reasoning が出力枠を食い切る。fail ではなく「答えていない」） |
| qwen3.8-flash-next-whitehacker | **error research-service-unavailable**（lane 停止中、538 s timeout） |
| qwen3.8-27b-uncensored（k16、2.5 tok/s） | 未測定（module 1 本 = 数千 token、1 task 30 分超） |

つまり **今日 fleet に routing されている model は 1 問も答えられていない**。fail ではなく truncated / error で
数えられていることが今回の harness の価値（8 問中 0 answered を「0 pass」と読まない）。

### 段 1: Hermes transcript → typed decision（`hermes_data.py`）
`~/.hermes/**/state.db`（282 profile、default だけで 144,686 message / 70,479 tool call、amu-maint だけで 11,698
call）を read-only で読み、assistant の tool call ごとに 2 判断: Choice「次に呼ぶ tool は」（option = その session が
使った tool）と Noul「この call はエラー無く終わるか」（terminal は `exit_code`、他は error marker）。state は
直前の user 発話と直近 2 tool 結果（同 turn の assistant 文は名指しの漏れなので入れない）。**secret scrub**（bearer /
`sk-` / `TOKEN=` / 長い hex・base64 を `[REDACTED]`、scrub 後も残る行は捨てる）を通し、出力は `data-hermes/`
（gitignore）。合成 DB での test 緑。**本物の DB への実行は agent の権限で拒否された**（transcript の一括読み出し =
provenance）。owner が回すなら:
```
cd orgs/kotoba-lang/typed-decisions && .venv/bin/python -m typed_decisions.hermes_data --out data-hermes
```
出力の `summary.json` に「検証付き triple の数」と profile 別 terminal 成功率が出る —— 成長速度の指標の初値。

## 公開 model（2026-09-18）

**https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large**（Apache-2.0、public）—— DeBERTa-v3-large、
augment p=0.7、seed 2、1 ep。in-domain **0.854** / ECE 0.022、OOD **0.690** / ECE 0.035（`reports/pub-open-jev-deberta-v3-large-*.json`）。
bundle = backbone（HF 形式）+ `head.safetensors` + marker 付き tokenizer + `open_jev_config.json`（pool / 温度 / 学習来歴 /
実測値）+ `typed_decisions/{schema,encoder,open_jev}.py`（loader を同梱、`pip install` 無しで動く）。Hub からの
round trip（clean dir に snapshot → `OpenJev.from_pretrained` → decide）を確認。M1 Max CPU fp32 で 4 question 1.8 s。

```python
from typed_decisions.open_jev import OpenJev
m = OpenJev.from_pretrained("com-kotobalabs/open-jev-deberta-v3-large")
m.decide(state, [{"type": "choice", "instructions": "...", "options": [...]}, {"type": "score", ...}, {"type": "noul", "instructions": "..."}])
```

ModernBERT-base 版（`pub-open-jev-modernbert-base`、augment 0.7、2 ep）は **0.504 に崩れた**（augment 無し 2 ep は 0.728）——
ModernBERT-base の run 間不安定の 4 例目。公開しない。

## コミット 1（2026-09-18）: 幅（16 family）と question 側の多様性 —— OOD の見積りは外れた

「公開 dataset 10 種で OOD 0.69 → 0.75」という見立てを 2 つの実験で測った。どちらも DeBERTa-v3-large、
augment 0.7、1 ep。

**A. 幅（`data_multi.py`、13 dataset × 3k state = MNLI / RTE / QNLI / MRPC / PAWS / CoLA / emotion / CLINC 20-way /
AG News / DBpedia / Yelp stars / IMDB / counterfactual、計 16 family、56k state / 92k question、$0.70）**

| | 元 3 source の in-domain | 新 13 family の in-domain | **OOD（固定 set）** |
|---|---|---|---|
| 3 family（公開版） | 0.854 / ECE 0.022 | — | **0.690** |
| 16 family seed 2 | **0.865 / ECE 0.009** | 0.78〜0.99（MNLI 0.87、PAWS 0.93、DBpedia 0.99、CoLA 0.78、Yelp 0.79） | 0.669 |
| 16 family seed 0 | 0.860 / ECE 0.014 | — | 0.670 |

**B. 同じ state に question 側の多様性（`data.py --extra-train-families`、元 3 source の train state に別の規則 gold
family を 2〜3 問ずつ追加、42k → 90k question、$0.47。test / OOD は全 3,000 state で比較）**

| | in-domain | OOD | b77 outflow | boolq support | sst5 score |
|---|---|---|---|---|---|
| baseline（全 test） | 0.860 | 0.667 | 0.74 | 0.60 | 0.46 |
| + train families | 0.860 | **0.670** | 0.81 | 0.75 | 0.37 |

結論: **どちらも OOD 合計を動かさない**（±0.5 pt、seed spread 0.025 の中）。family 別には ±10 pt 動くが方向が
ばらばらで、合計では相殺する。幅は in-domain と較正（ECE 0.022 → 0.009）を上げ、新 family はどれも訓練に
入れば 0.78〜0.99 になる —— **「訓練に入れた question は読める、入れていない question は 0.67 前後」**が
この model class（435M encoder、1 forward）の今の形で、私の「5 コミットで 0.8」は外れ。正しい見立ては下の
「見積りの訂正」。

16 family 版は HF の同 repo に **revision `multi-16`** として公開（card にこの比較を書いた）。`main` は 3 family 版のまま。

### 見積りの訂正
- **あなたの question family を訓練に入れる**: 1 family あたり数千 state の gold（または教師ラベル）で 1 コミット、
  in-domain 0.8〜0.99 に届く（今日 13 family が全部そうなった）。これが「精度が高くなる」の実際の経路。
- **未見の question を読む力（OOD 合計）**: 今日試した 2 lever では動かない。動かす候補は backbone を 1 段上げる
  （fleet に載る architecture で）か、OOD family 自体に教師ラベルを付けて in-domain 化するか。後者は結局 1 と同じ。

## 第5反復（2026-09-19）: 新 family `repo-governance` —— 実 Jev API を workspace 自身の判断に向けた最初の記録

owner の問い: workspace 内の agent loop（superproject の repo-bot / detector が見つける finding）を jev 形の
Choice/Score/Noul で処理し、あとで学習に使える dataset として記録・公開できるか。ここまでの family（banking77 /
sst5 / boolq、code_data、kotoba-tasks、hermes_data）は全て **外部 gold か runnable test を持つ**。今回追加した
`repo-governance` はどちらも持たない —— superproject の finding（例: compliance-scope-boundary の
cross-boundary finding）に「正解」を決めている者がまだいない、未解決の運用判断だから。

**やったこと。** superproject 側の新ツール `scripts/jev-decide.cljk`(kotoba-lang/com-junkawasaki@codex/jev-decision-probe)
から、実際の TypeSafe Jev（`typesafe/jev-1.13`、OpenRouter `POST /api/alpha/decisions`、この repo の学習済み
model とは別物 —— 教師でも比較対象でもなく、記録対象の予測器）を呼び、その `state + questions + answers` を
`data/repo-governance.jsonl` に 1 行 1 example で追記する。schema は `schema.Example` を再利用しつつ、
**`Question.gold` を全問 `null` にし**、jev の実際の答えは `gold` にではなく別フィールド `prediction` に置く
（このリポジトリの一貫した方針 ——「教師ラベルは gold の代用にしない」—— を、外部 gold が最初から無い family
にも適用しただけ）。`gold_status` フィールドに `"unresolved -- ..."` と明記し、この family を読む側が
`prediction` を gold と取り違えないようにした。

**現状は corpus ではなく種（n=1）。** 1 件目（`meta.finding_id = "cross-boundary:cloud-itonami-kaisya"`）を記録:
`needs_urgent_remediation` noul 0.42、`owning_area` choice `kotobase-planes`（p=0.80, confidence 0.73）、
`severity` score 1.05/2（"moderate"、confidence 0.91）。cost $0.0000275/call。**訓練にはまだ使えない** ——
`gold` が埋まって初めて他 family と同じ扱いになる。埋める経路は 2 つ、どちらも未着手: (a) owner か
governor が実際にこの finding を裁定した時点でその裁定を `gold` に書き戻す、(b) 十分件数が溜まったら
`teacher.py` と同じ形で外部の強い model に多数決させ、教師ラベルとして明示区別する（`prediction` を
増やすだけで `gold` にはしない）。

**公開。** `data/repo-governance.jsonl` を新規 HF dataset `com-kotobalabs/typed-decisions-repo-governance`
として Apache-2.0 で公開（owner 指示 2026-09-19、内容の compliance finding をそのまま含めてよいと確認済み）。
card に「ungoaled、prediction フィールドは jev の予測であって正解ではない」ことを明記。他 family のような
train/val/test 分割・精度表はまだ無い —— n=1 の種であることを card にもそのまま書く。

## 第6反復（2026-09-19）: choice しか返さない model で kotoba の refactor は回るか —— `jev_holes.py`

owner の問い: 「jev は 1 回の呼び出しでコードを書けない」は per-call の話で、生成とは反復された choice のこと。
駆動ループ（穴を作る）・候補列挙（有限の option）・verifier（kbb で test）を外から与えれば、実 Jev で
kotoba の fix pair を通せるか。test bed は段 0 の `data/kotoba-tasks.json`（24 pair、FAIL→PASS 検証済み）。

**手順（`src/typed_decisions/jev_holes.py`、model は書かない）。** before/after を token 差分し、**1 token 削除 →
1 token 挿入、両方 code-like**（keyword / symbol、文字列と `;` コメントは除外）の hunk だけを *穴* とする。option は
「壊れた module と test に出る同種の token」（test が仕様であり、module が提供すべき名前を挙げている）**から
置換前の token を除いたもの**。gold が option に無ければ *unreachable* として記録する（足さない）。穴ごとに
Jev へ `choice` を 1 回（state = commit message + test + `<<HOLE>>` 入り module + 修正前の test 出力）、独立に。
全穴に top-1 を入れて kbb で実 test、FAIL なら confidence の低い穴から Jev 自身の分布の次点へ backtrack、
1 task 8 run まで。

**corpus の形が最初の結果。** 変更 token は挿入 2,453 / 削除 191 —— **93% が新規挿入**で、1 token 置換の穴は
51 個（うち 37 個は 1 task の `:inject`→`:no` の繰り返し）、穴を 1 つでも持つ pair は 24 中 6、**穴だけで
直る pair は 1 つ**（`kototama-beae237c`）。この corpus は「test が反転する bugfix」を掘ったものなので生成に
偏るのは当然で、dependency-substitution のような繋ぎ替え refactor は test を反転させないため入っていない。
それでも「実 bugfix の大半は choice の外」は数字として残す。

**穴の精度（run 2、置換前 token を除外後、45 穴、mean chance 0.008 ≈ 1/122）。**

| | 穴 | top-1 | top-3 | MRR |
|---|---|---|---|---|
| 全体 | 45 | **0.889** | 0.956 | 0.924 |
| 契約名への rename（`:log-append!`→`:log-write`、`:now`→`:clock-monotonic`、`update`→`update-in`、`:inject`→`:no`×37） | 40 | **1.000** | — | — |
| 論理の変更（`second`→`first`、`init`→`n`、`pm/leader-for-view`→`c/leader-for`） | 5 | **0.000** | 0.6（rank 2 が 3 件） | — |
| unreachable（`names` / `live` の新規識別子、`:max-log-write-bytes` は test にも module にも無い） | 6 | 測れない | | |

**較正が最も強い結果。** run 1（置換前 token を option に残した）では誤答 6 件中 4 件が **「変えない」を
confidence 0.89〜0.93 で選ぶ**現状維持バイアスで、正誤の confidence は重なった（正 0.76 / 誤 0.69）。除外した
run 2 では **正答 min 0.79 / 誤答 max 0.56** と完全に分離 —— 閾値 0.6 で **auto-apply 40/40 正、escalate 5/5 誤**。
ADR-2609181715 の `:autonomous` / `:escalate` 閾値がそのまま機能する形。

**end-to-end。** 6 task 中 PASS 1（`beae237c`、greedy 1 run）。残りが落ちる理由は model ではない: `de177068` は
到達可能な 2 穴を両方正解したが `:max-log-write-bytes` が spec にも file にも無い（commit message が言う
`kotoba-core-contracts` を候補源に足せば届く —— **候補列挙器の問題**）、`inga` / `engi` は穴以外の挿入・削除
hunk を持つ（**穴だけでは直らない pair**）、`native` / `sema` は新規識別子 2 つ（**生成**）。cost $0.018 / 45 穴
（state が 4〜6k token なので $0.0004/穴、`$0.00003/呼び出し` の見積りは短い state の値）、45 API + 41 kbb run
で壁時計 29 s。

**読み方。** 「jev で refactor」は「できる／できない」ではなく 3 層に分かれる: (1) **既存の名前への繋ぎ替え**は
option に名前がありさえすれば 40/40、較正込みで自動適用できる。(2) **論理の選択**（どの変数・どの関数）は
top-1 0/5、rank 2 が多いので backtracking 付き探索の入口にはなるが単独では信用できない。(3) **新規の名前・
新規の式**は option に無いので choice の外。この corpus では変更 token の 93% が (3)。次に測るのは候補源を
symbol-index / kotoba-core-contracts に広げたときの reachable 率（`de177068` が通るか）と、(2) を型で刈ったときの
rank（ADR の未着手項目そのもの）。

## 第7反復（2026-09-19）: 候補源を repo に広げ、役割で刈る —— `jev_holes.py --pool repo --prune-role`

第 6 反復が「model ではなく候補列挙器の問題」と名指しした 2 つを測った。(a) 候補源を module + test から
**sha 時点の repo 全 source**（symbol-index が返すもの）に広げたときの到達率、(b) 穴の**構文上の役割**
（`(` 直後 = call / binder の vector 内 = binding / keyword / それ以外 = arg）で候補を刈ったときの rank。
(b) は型による刈り込みの *代理* —— これらの pair は `.cljc` で、kotoba-sema が型を付けるのは `.kotoba` だけ。

**先に harness の欠陥を 1 つ直した。** 文字列 literal を丸ごと除外していたが、空白を含まない文字列は
docstring ではなく *名前*（wire 名、model id）で、`de177068` の `"log-append!"`→`"log-write"` /
`"now"`→`"clock-monotonic"` と `app-kotoba-cloud` の model id 3 件がこれに当たる。`str` kind として穴に数え、
候補には同種の文字列に加えて **pool 内の keyword の名前を文字列にしたもの**を足す（名前文字列は隣の keyword と
対になるのが普通）。穴は 51 → 56。

| arm | 到達 | top-1 | top-3 | 正答 conf min | 誤答 conf max | 閾値 0.6: 自動適用 正/誤 · escalate | e2e |
|---|---|---|---|---|---|---|---|
| file | 47/56 | 0.894 | 0.957 | 0.77 | 0.51 | 42 / 0 · 5 | 2/7 |
| file + role | 43/56 | **1.000** | 1.000 | 0.78 | — | 43 / 0 · 0 | 2/7 |
| repo | **55/56** | 0.855 | 0.891 | 0.37 | **0.79** | 46 / **2** · 7 | 2/7 |
| repo + role | 55/56 | 0.891 | 0.891 | 0.36 | 0.79 | 47 / 1 · 7 | 2/7 |

**読み方。**
- **rename refactor が 1 本、choice だけで通った。** `kototama-de177068`（keyword 2 + 名前文字列 2 を
  `kotoba-core-contracts` の名前に揃える）は 4 arm すべてで greedy 1 run PASS、6 穴中 6 正解。file pool でも
  通る —— `:max-log-write-bytes` の 2 穴は test に不要だった。第 6 反復で落ちていたのは文字列の穴を
  埋めていなかったからで、到達率ではない。
- **repo pool は到達を 47 → 55 にする**（残る 1 つ `live` は 255 上限の 3-gram 選抜で落ちた）。新たに届いた
  `:max-log-write-bytes` ×2 と model id 文字列 ×3 は **5/5 top-1**。ただし代償があり、論理の穴の distractor が
  増えて **較正が崩れる**: `init`→`n` に `initial`（0.79）、`pm/leader-for-view`→`c/leader-for` に
  `leader-for-view`（0.75）を *自信を持って* 選ぶ。3-gram で 255 に絞る私の cap が、まさに似た名前を
  選んで残すので、model が使っている名前の類似性という手掛かりを裏返しに突く。file pool では正誤の confidence が
  分離していた（min 0.77 / max 0.51）のが、repo pool では逆転する（0.37 / 0.79）。**閾値は pool に依存する。**
- **役割で刈るのは効くが、代理は代理。** call 位置の穴 `c/leader-for` は role を付けると全 arm で top-1
  （接頭辞の無い `leader-for-view` が消える）、`first` も repo + role で一度 top-1。一方 file + role の 1.000 は
  **生き残った 43 穴の上の数字**で、刈り込みが gold を 4 つ落としている（`first` は値として使われるが file
  内では call 位置にしか現れない、`n` は binding として現れない）—— 型なら落とさない。binding 名の穴
  （`init`→`n`）は役割で刈っても 522〜794 候補が残り rank 49〜171: **局所変数の命名は候補列挙では解けない。**
- **e2e は 2/7 で頭打ち。** 残り 5 の理由は model の外: `native` / `sema` は新規識別子（`names` / `live`）+
  論理、`inga` / `engi` / `app-cloud` は穴以外の挿入・削除 hunk を持つ。

**結論の更新。** 「既存の名前への繋ぎ替え」は候補源が repo なら **到達 55/56、正答 46〜47/50** で、rename
refactor は 1 本通る。候補を広げるほど論理の穴の較正が悪化するので、`:autonomous` の閾値は **穴の種類
（keyword / 名前文字列 / call 位置 / binding）ごと**に持つのが正しく、単一の 0.6 ではない。論理の穴と
新規の名前は引き続き choice の外。次に効くのは (1) この 56 穴を **gold 付き family として corpus に入れる**
（第 5 反復の repo-governance と違い、gold は commit と test が保証する）、(2) 3-gram ではなく型・役割で
255 に絞る cap、(3) `.kotoba` の pair が溜まったら kotoba-sema で刈って同じ表を取り直す。

## 第8反復（2026-09-20）: `code-holes` family —— git 履歴から gold 付きの穴を掘り、jev を held-out で測る

第 7 反復の次の一手 (1): 56 穴ではなく、**kotoba-lang の git 履歴そのもの**から 1 token 置換を掘って gold 付き
family にする（`hole_data.py`）。gold は commit が置いた token —— 教師ラベルより強く、test 検証済み pair より
弱い（commit は間違えうる、test は偶然通らない）。`verified: "commit" | "test"` で区別。option は第 6 反復と同じ
「module + test にある同種 token − 置換前」、gold が無ければ落として数える。split は **repository 単位**
（test の穴は train に無い repo から）。

**掘れたもの（80 repo × 300 commit、`reports/hole-data-stats.json`）。** file diff 5,941、穴 4,042、
**gold 到達 1,300（symbol 969 / keyword 283 / str 48）、unreachable 2,742 = 68%** —— rename 先は編集中の file には
無いのが普通で、第 7 反復の「候補源を repo に」はここでも効く（未測定）。挿入 1.31M / 削除 221k token、
86% が挿入。

**1 回目の held-out 評価は corpus の質を測っていた。** test split 200 穴で top-1 0.225（keyword 0.82、str 0.63、
**symbol 0.08**）。内訳を見ると **200 穴中 167 が「変更 token 100 超の file diff の中の 1:1 alignment」**
—— 大きな書き換えの中で偶然 1 対 1 に並んだ token で、誰も決めていない置換（`mapv`→`dec`、
`brace-close`→`body`）。そこで symbol 0.05。11〜100 token の範囲では keyword/str **14/14**、symbol 5/16。
各 record の meta に `changed_tokens` / `holes_in_file` を持たせて掘り直し、**isolated（≤100）は 1,300 中 344**。

**isolated 344 穴で jev（`reports/hole-eval-jev-iso100.json`、$0.13、112 s）。** jev は何も学習していないので
split は無関係、全 split の isolated を使う。

| 変更 token | n | 全体 | keyword | str | symbol |
|---|---|---|---|---|---|
| 0–10 | 56 | 0.61 | 16/19 = 0.84 | 2/4 | 16/33 = 0.48 |
| 11–30 | 92 | 0.50 | 26/47 = 0.55 | 11/16 = 0.69 | 9/29 = 0.31 |
| 31–100 | 196 | 0.57 | 83/117 = 0.71 | 2/10 | 27/69 = 0.39 |
| **合計** | **344** | **0.558** | **0.683**（chance 0.013） | 0.50 | **0.397**（chance 0.006） |

閾値ゲート（自動適用 正/誤 · escalate）: 0.6 → 142/33 · 169、0.7 → 122/17 · 205、**0.8 → 108/9 · 227**
（精度 0.92、適用率 0.34）。正答 confidence 平均 0.75 / 誤答 0.42。

**読み方。**
- **第 6・7 反復の 0.89 は keyword に偏った 56 穴の数字**（`:inject`→`:no` ×37）。履歴全体では keyword 0.68、
  symbol 0.40 が jev の実力で、chance の 50〜60 倍だが半分は外す。
- symbol で当たるのは **依存の繋ぎ替え**そのもの: `clojure.string`→`kotoba.lang.text`、
  `clojure.data.json`→`json.data-json`、`commit-dag.core`→`chain.core`、`seq`→`empty?`、`update`→`update-in`
  —— dependency-substitution-wave が数千件単位で抱えている形。外すのは **演算子・述語の意味変更**
  （`inc`→`quot` に `+` を 0.94、`>`→`=` に `not=` を 0.83）と **接頭辞だけの lookalike**（`g/add-edge`→
  `g/add-conditional-edges` に `g` を 0.98）。後者は候補の整形（名前空間付き symbol の穴に裸の alias を
  出さない）で消せる。
- 0.8 ゲートは「3 割を精度 9 割で自動、7 割は escalate」。escalate 先が LLM か人かは別の話だが、
  **人が 1,300 件を全部見る代わりに 227 件を見る**のがこの model の今の価値。
- 未測定: repo pool での到達（2,742 の unreachable がどこまで届くか）、student（DeBERTa）をこの family で
  訓練したときの in-domain / OOD、candidate の整形後の symbol 精度。

**公開（2026-09-20）。** owner が kotoba-lang は public と確認、**GitHub API で 65 repo の visibility を実測したら
64 public / 1 private（`app-kotoba-cloud`、68 record）** —— private の record を除いた 1,232 件（train 877 / val 70 /
test 285、isolated ≤100 は 273 / 18 / 37）を `com-kotobalabs/typed-decisions-code-holes` として Apache-2.0 で公開。
card に `changed_tokens` で filter せよと書き、jev の isolated 344 の数字を載せた。corpus は `hole_data.py` で
再生成できる（data は commit しない慣例のまま）。
## 第9反復（2026-09-20）: 候補の整形と repo pool —— 到達は 3 倍、symbol の精度はその分だけ落ちる

第 8 反復の次の一手 (3) と (1)。どちらも `hole_eval.py --reshape` / `hole_data.py --pool repo`。

**(3) 候補の整形（`reshape`）。** 穴の置換になり得ない候補だけを落とす: reader 構文（`#` `'` `` ` `` `~` `@` `^`
始まり）は常に、**namespace alias の裸 symbol**（`g/add-edge` の穴に `g`）は穴が namespaced なときだけ、`_` / `&` は
穴が binding 位置でないときだけ。最初の版は無条件に落として gold を 4 つ消した（`id`→`_` は unused binding の
実 refactor、`ui`→`dds-tokens` は alias が裸 symbol を置き換える実例）ので、穴の形で条件付けた。isolated 344 穴で
候補 136 個を落とし gold は 0。**再採点: 0.558 → 0.567、symbol 0.397 → 0.412。** 穴ごとの反転は 正→誤 2 / 誤→正 7、
うち整形が実際に候補を変えた穴での反転は 4（`clojure.data.json`→`json.data-json` ×3 で `json` が消えて正解）—— 残り
5 は候補が同じ穴で答えが変わった、つまり **jev の呼び出し間の揺れが ±1.5 pt ある**。整形の効果はその揺れと同程度。

**(1) repo pool（20 repo × 150 commit、blob を cache して sha 時点の全 source を候補源に）。**

| | 穴 | 割合 |
|---|---|---|
| 見つかった穴 | 1,957 | |
| file pool で到達 | 425 | 22% |
| repo pool で新たに到達 | 833 | +43% → **64%** |
| なお unreachable（新規の名前） | 699 | 36% |

isolated（≤100）355 穴で jev（整形あり、cap 255 は役割優先）:

| 到達源 | n | 全体 | keyword | str | symbol | mean k |
|---|---|---|---|---|---|---|
| file でも届く | 147 | 0.67 | 86/110 = 0.78 | 2/3 | 10/34 = 0.29 | 246 |
| **repo でしか届かない** | 208 | 0.34 | 24/33 = **0.73** | 18/26 = **0.69** | 29/149 = 0.19 | 248 |
| 合計 | 355 | 0.48 | 0.77 | 0.69 | 0.21 | 247 |

閾値 0.8: 自動適用 97 件中 83 正 / 14 誤（精度 0.86、適用率 0.27）、258 escalate。

同じ穴で pool だけ変えた対照（両 run に共通の 53 穴）: 全体 0.51 → 0.47、keyword 15 → 16、**symbol 10 → 7**、
mean k 148 → 242。**候補を 3 倍に広げても keyword は落ちず、symbol が 3/26 落ちる** —— 第 7 反復と同じ向き。

**読み方。**
- **repo pool は明確に得。** 到達 22 → 64% で、新たに届いた穴のうち keyword 0.73 / 名前文字列 0.69 は file pool と
  同水準。symbol 0.19 の中身は依存の繋ぎ替え（`clojure.string`→`kotoba.lang.text`、`clojure.edn`→`kotoba.lang.edn`、
  `multiformats.core`→`sha2.core`、`hiccup/->html`→`html/html5`）—— **rename 先が別の file に既に在る、という
  dependency-substitution の形そのもの**が、この model が最も得意な穴。
- 残る 36% は sha 時点の repo に無い名前 = 生成。ここは choice の外で、変わらない。
- symbol は候補が増えるほど落ちる（0.38 → 0.27）。cap 255 に対して mean k 247 なので、255 の中で **役割・型で
  さらに刈る**か、jev の分布の top-k を第二段の判定（型検査、test）に渡す設計が要る。
- 公開: この run の repo-pool 版は `com-kotobalabs/typed-decisions-code-holes` に config `repo-pool` として追加
  （private の `app-kotoba-cloud` を除く）。
