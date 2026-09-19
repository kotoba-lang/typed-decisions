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
