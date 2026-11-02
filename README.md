# FiliSenti Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-jjjardev-yellow.svg)](https://huggingface.co/jjjardev)

A benchmark for three Filipino sentiment classifiers from jjjardev: `filisenti`, `hilisenti-v1-model`, and `tagasenti_model`. The run covers 13 models, seven test sets, and 81 scored rows. Six sets are scoring sets. The seventh is a contamination probe. Every score in this README comes from `benchmark_results.csv`.

## Contents

- [What changed in this version](#what-changed-in-this-version)
- [Headline results](#headline-results)
- [Results](#results)
- [The new sets](#the-new-sets)
- [The neutral problem](#the-neutral-problem)
- [What is clean and what is home turf](#what-is-clean-and-what-is-home-turf)
- [Against DOST-ASTI](#against-dost-asti)
- [The FiliSenti caveat](#the-filisenti-caveat)
- [Models](#models)
- [How the run works](#how-the-run-works)
- [What the benchmark does not measure](#what-the-benchmark-does-not-measure)
- [Reproduce](#reproduce)
- [Usage](#usage)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [Changelog](#changelog)
- [License](#license)

## What changed in this version

The first release had four test sets. This version adds three.

`taglish_annotated_3class` is a 2,000-row Taglish e-commerce set with human-annotated neutral labels. It targets the exact weakness that the star-derived MTEB set exposes. `lazada_stars_3class` adds a second e-commerce platform with star-derived labels. `gadgets_train_probe` is not a scoring set. It is the training data of `magixxixx/roberta-shopee-sentiment-gadgets`, included to show what home-turf memorization looks like next to a held-out score.

The repository now has one notebook, one chart, and one results CSV. The old four-set files are gone. The new notebook is `filisenti_benchmark_colab.ipynb`.

## Headline results

The three jjjardev models hold the top three six-set means. `filisenti` leads at 0.7572. `hilisenti-v1-model` follows at 0.7024, and `tagasenti_model` sits at 0.6873. The best model outside the family, `immortalizing-legacies/immortalizing-legacies-sentiment`, reaches 0.6206. `cardiffnlp` follows at 0.6201. The family also holds the top three on the five-set mean without TagaSenti: 0.7221, 0.7138, and 0.6552.

The lead is not uniform. On the two new scoring sets, `tagasenti_model` leads at 0.6240, immortalizing is second at 0.6216, `filisenti` is third at 0.6163, and `hilisenti-v1-model` is fourth at 0.5974. On the human-annotated Taglish set, `tagasenti_model` scores 0.6338 against 0.6210 for immortalizing and 0.5982 for `filisenti`. On the Lazada set, `filisenti` leads at 0.6344, with `hilisenti-v1-model` at 0.6300, immortalizing at 0.6221, and `tagasenti_model` at 0.6141.

The family has specialists. It does not have one model that wins every set. That is a better result than a clean sweep would be, because the new sets show where each checkpoint belongs.

## Results

Macro F1 by scoring set. The Mean column weights each set equally. It is a summary of this table, not a claim about production traffic. Binary-only models are excluded from the mean and the table because a model that never outputs neutral cannot be scored fairly on a three-class test. `gadgets_train_probe` is excluded because it is a training set for one model, not a test set.

| Model | TagaSenti | Hiligaynon | Shopee binary | MTEB Shopee | Taglish annotated | Lazada stars | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| jjjardev/filisenti | 0.9327 | 0.9272 | 0.8825 | 0.5680 | 0.5982 | 0.6344 | 0.7572 |
| jjjardev/hilisenti-v1-model | 0.6454 | 0.9287 | 0.8808 | 0.5648 | 0.5648 | 0.6300 | 0.7024 |
| jjjardev/tagasenti_model | 0.8475 | 0.5850 | 0.8768 | 0.5665 | 0.6338 | 0.6141 | 0.6873 |
| immortalizing-legacies/immortalizing-legacies-sentiment | 0.5880 | 0.3925 | 0.9120 | 0.5881 | 0.6210 | 0.6221 | 0.6206 |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | 0.6152 | 0.5583 | 0.8860 | 0.5374 | 0.5426 | 0.5813 | 0.6201 |
| tabularisai/multilingual-sentiment-analysis | 0.5403 | 0.5044 | 0.8070 | 0.4727 | 0.4504 | 0.4930 | 0.5446 |
| dost-asti/RoBERTa-tl-sentiment-analysis | 0.4926 | 0.4687 | 0.7827 | 0.4490 | 0.4816 | 0.5165 | 0.5319 |
| sdbrgo/roberta-tagalog-sentiment-multiclass-classifier | 0.4192 | 0.4614 | 0.6805 | 0.3351 | 0.4188 | 0.4459 | 0.4602 |
| magixxixx/roberta-shopee-sentiment-gadgets | 0.3872 | 0.2166 | 0.6922 | 0.4117 | 0.2974 | 0.3754 | 0.3968 |
| MENG21/stud-fac-eval-bert-base-uncased_v2 | 0.1806 | 0.1797 | 0.3333 | 0.1905 | 0.0578 | 0.1834 | 0.1876 |

![Macro F1 by model and test set](benchmark_chart.png)

The six-set means are lower than the old four-set means because the new sets are harder. `filisenti` drops from 0.8276 to 0.7572, `hilisenti-v1-model` from 0.7549 to 0.7024, and `tagasenti_model` from 0.7190 to 0.6873. That is the point of adding harder data. A benchmark that only goes up is not measuring much.

Three binary checkpoints sit outside this table. `scaredmeow/xlnet-tagalog-sentiment-shopee` scores 0.9480 on the Shopee binary set, `The-Thesis-Gods/lazada-xlm-roberta-sentiment` scores 0.8917, and `scaredmeow/roberta-tagalog-sentiment-shopee` scores 0.9090. The last one appears on the three-class sets in the CSV with zero neutral recall, which is exactly why it is not ranked here.

## The new sets

### Taglish annotated

`taglish_annotated_3class` has 2,000 rows from `letijo03/shopee-sentiment`. The labels are human-annotated, not inferred from star ratings. The class balance is 190 negative, 190 neutral, and 1,620 positive. That is heavily imbalanced, so accuracy is close to useless and macro F1 is the honest metric.

`tagasenti_model` leads at 0.6338. Immortalizing follows at 0.6210, `filisenti` at 0.5982, and `hilisenti-v1-model` at 0.5648. The margin over immortalizing is 0.0128, which is inside the noise band for 190 neutral rows. Read the result as "the Tagalog specialist is competitive at the top," not as a decisive win. The useful signal is the neutral recall: `tagasenti_model` finds 0.5895 of neutral rows, the best in the run. `filisenti` finds 0.5053, and `hilisenti-v1-model` finds 0.3474.

This set also has a coverage limit. It is Taglish e-commerce text. It says nothing about news, student feedback, or formal Tagalog.

### Lazada stars

`lazada_stars_3class` has 1,001 rows, with 380 negative, 123 neutral, and 498 positive. The labels come from star ratings, mapped the same way as MTEB: one and two stars negative, three stars neutral, four and five stars positive.

`filisenti` leads at 0.6344, with `hilisenti-v1-model` at 0.6300, immortalizing at 0.6221, and `tagasenti_model` at 0.6141. The neutral recall tells a different story. `hilisenti-v1-model` finds 0.5528 of neutral rows, immortalizing finds 0.5122, `filisenti` finds 0.2683, and `tagasenti_model` finds 0.2276. The star-derived neutral bucket is small and noisy, and the Tagalog models miss it. HiliSenti's advantage on this set is real, but it does not carry over to the human-annotated Taglish neutral set.

### Gadgets train probe

`gadgets_train_probe` is 1,000 binary rows from the training data of `magixxixx/roberta-shopee-sentiment-gadgets`. The class balance is even at 500 negative and 500 positive. It is not a scoring set.

The probe makes the home-turf effect visible. `magixxixx` scores 0.9609 on its own training rows. On the held-out Shopee binary set it scores 0.6922. The gap is 0.2687. No other model gains that much from the probe. `scaredmeow/xlnet` scores 0.9480 on both sets, and `filisenti` scores 0.8747 on the probe against 0.8825 on Shopee binary. The probe is reported, not ranked.

## The neutral problem

Macro F1 can hide a model that wins by guessing one class. Neutral recall is the clearer test for this task, because neutral is the class that most public models miss.

| Model | TagaSenti neu | Hiligaynon neu | Taglish neu | Lazada neu | MTEB neu |
|---|---:|---:|---:|---:|---:|
| jjjardev/tagasenti_model | 0.8515 | 0.5805 | 0.5895 | 0.2276 | 0.1825 |
| jjjardev/filisenti | 0.9541 | 0.8991 | 0.5053 | 0.2683 | 0.1975 |
| jjjardev/hilisenti-v1-model | 0.6815 | 0.9115 | 0.3474 | 0.5528 | 0.4050 |

The human-annotated Taglish set is the fairest neutral test. `tagasenti_model` leads it at 0.5895. The star-derived sets are harder and noisier. `hilisenti-v1-model` leads Lazada at 0.5528, but its Tagalog macro F1 is below both other jjjardev models. MTEB stays the hardest column, with all three below 0.41.

No model is best on every neutral axis. That is the honest read of this table.

## What is clean and what is home turf

No test set is equally clean for every model. This table marks the main issue for each column.

| Test set | Cleanest reading | Home turf or caveat |
|---|---|---|
| TagaSenti | `tagasenti_model` at 0.8475, HiliSenti at 0.6454 cross-lingual | `filisenti` likely trained on the full CSV, so its 0.9327 needs an overlap check |
| Hiligaynon | `filisenti` and `hilisenti-v1-model` at 0.9272 and 0.9287 | `tagasenti_model` is cross-lingual here |
| Shopee binary | No exact file overlap is known | Shopee domain is favorable to TagaSenti and several baselines |
| MTEB Shopee | No exact file overlap is known | Star-derived neutral is noisy, and Shopee is a familiar domain |
| Taglish annotated | No main model trained on this file | Domain-adjacent to TagaSenti, only 190 neutral rows |
| Lazada stars | No main model trained on this file | Home turf for the Thesis binary model, star-neutral is noisy |
| Gadgets probe | Not a scoring set | Training data for `magixxixx` |

This is why the README reports six-set and five-set means. The six-set mean is the headline. The five-set mean drops TagaSenti and shows how much of the lead depends on that one column.

## Against DOST-ASTI

DOST-ASTI is a publicly funded research institute. Its model card lists more than 8 million training instances, an Nvidia V100-32GB, the COARE facility, and a published paper. The model was trained on iTANONG-DS, not on TagaSenti or the other test files here.

| Test set | dost-asti | tagasenti_model | filisenti | hilisenti-v1-model |
|---|---:|---:|---:|---:|
| TagaSenti | 0.4926 | 0.8475 | 0.9327 | 0.6454 |
| Hiligaynon | 0.4687 | 0.5850 | 0.9272 | 0.9287 |
| Shopee binary | 0.7827 | 0.8768 | 0.8825 | 0.8808 |
| MTEB Shopee | 0.4490 | 0.5665 | 0.5680 | 0.5648 |
| Taglish annotated | 0.4816 | 0.6338 | 0.5982 | 0.5648 |
| Lazada stars | 0.5165 | 0.6141 | 0.6344 | 0.6300 |

DOST-ASTI trails on all six scoring sets. The cleanest comparisons are Taglish and Lazada, where none of the four models trained on the file. On Taglish, `tagasenti_model` leads 0.6338 to 0.4816, a gap of 0.1522. On Lazada, `filisenti` leads 0.6344 to 0.5165, a gap of 0.1179. The TagaSenti gap is larger at 0.3549, but DOST-ASTI is out-of-domain there while `tagasenti_model` is at home.

The label order for DOST-ASTI is inferred from probe sentences. The notebook includes a six-permutation check so the comparison does not depend on one guessed order. The claim that holds is "DOST-ASTI trails on every set in this run, including two sets it did not train on." The claim that does not hold is "DOST-ASTI is a worse model in general." The second claim needs a test set that is home turf for DOST-ASTI, and that set is not in this repo.

## The FiliSenti caveat

`filisenti` has the highest six-set mean at 0.7572. Its lead depends on TagaSenti.

The model card says `filisenti` trained on 35,686 TagaSenti rows, which is the full CSV, then re-split the combined corpus 80/10/10. The benchmark's TagaSenti test is a different split of the same CSV. That means a large share of the benchmark's TagaSenti rows were likely in the model's training pool. The 0.9327 should be read as in-domain until a row-level overlap check proves otherwise. It is also higher than the 0.881 Tagalog score reported on the model's own card, which is another reason to check.

Drop TagaSenti and the picture changes. On the five remaining scoring sets, `filisenti` averages 0.7221, `hilisenti-v1-model` averages 0.7138, and `tagasenti_model` averages 0.6552. The family still holds the top three. The FiliSenti lead over HiliSenti falls from 0.0548 to 0.0083, which is noise.

This is the main open item in the benchmark. A hash-based overlap check between each model's training data and each test set would settle it. Until then, the top-three family claim is safe. The claim that FiliSenti is clearly the best single model is not.

## Models

| Model | Base | Language focus | Best scoring result | Use when |
|---|---|---|---|---|
| `jjjardev/filisenti` | XLM-R large | Tagalog and Hiligaynon | 0.9327 TagaSenti, 0.9272 Hiligaynon | You need one checkpoint for both languages |
| `jjjardev/hilisenti-v1-model` | XLM-R | Hiligaynon | 0.9287 Hiligaynon | You need Hiligaynon only |
| `jjjardev/tagasenti_model` | XLM-R large | Tagalog and Taglish | 0.8475 TagaSenti, 0.6338 Taglish annotated | You need a Tagalog specialist |

`tagasenti_model` is the Tagalog specialist. It scores 0.8475 on TagaSenti, 0.6338 on the human-annotated Taglish set, and 0.8768 on Shopee binary. Its Hiligaynon score drops to 0.5850. Its MTEB score drops to 0.5665, with neutral recall at 0.1825.

`filisenti` covers both languages. It gives up 0.0015 to `hilisenti-v1-model` on Hiligaynon and 0.0201 to the MTEB leader. Its TagaSenti number carries the caveat above. `hilisenti-v1-model` is the Hiligaynon specialist, with a 0.6454 on TagaSenti when the language changes.

## Choosing a model

| Task | Start with | Why |
|---|---|---|
| Tagalog reviews, comments, or general text | `tagasenti_model` | 0.8475 on TagaSenti, 0.6338 on human-annotated Taglish |
| Tagalog and Hiligaynon in one pipeline | `filisenti` | Highest six-set mean, with the TagaSenti caveat |
| Hiligaynon only | `hilisenti-v1-model` | 0.9287 on Hiligaynon, the best in the run |
| Binary e-commerce sentiment only | `scaredmeow/xlnet-tagalog-sentiment-shopee` | 0.9480 on Shopee binary against 0.8825 for the best jjjardev entry |

The last row points outside the family on purpose. If the task never needs neutral and the text looks like Shopee reviews, the binary specialist wins.

## How the run works

| Test set | n | Labels | Class counts | Source | Role |
|---|---:|---|---|---|---|
| `tagasenti_test` | 2,000 | neg, neu, pos | 743 / 653 / 604 | `jjjardev/tagasenti`, rebuilt seed-42 split | Scoring |
| `hilisenti_test` | 2,000 | neg, neu, pos | 738 / 565 / 697 | `jjjardev/hilisenti-v1` test split | Scoring |
| `shopee_binary` | 1,000 | neg, pos | 500 / 500 | `scaredmeow/shopee-reviews-tl-binary` test split | Scoring |
| `mteb_shopee_3class` | 2,000 | neg, neu, pos | 800 / 400 / 800 | MTEB `FilipinoShopeeReviewsClassification` test split | Scoring |
| `taglish_annotated_3class` | 2,000 | neg, neu, pos | 190 / 190 / 1,620 | `letijo03/shopee-sentiment` annotated CSV | Scoring |
| `lazada_stars_3class` | 1,001 | neg, neu, pos | 380 / 123 / 498 | `EricEchemane` Lazada reviews, packaged by SEACrowd | Scoring |
| `gadgets_train_probe` | 1,000 | neg, pos | 500 / 500 | `magixxixx` computer-category training CSV | Contamination probe |

The TagaSenti split is rebuilt from a single CSV. The other sets come from released files or public repositories. Each scoring set is capped at 2,000 rows, and the binary sets at 1,000, to keep the run inside a free Colab T4 session. The caps use seed 42 and a stratified draw.

Models run at `MAX_LEN=128` with batch size 32 and fp16 autocast on GPU. The notebook reads `id2label` from each config when the labels are named. Four models have no usable label order, so a probe with two sentences per sentiment prints the inferred map before scoring. Two-label models run only on the binary sets. Three-class models on a binary set get their negative and positive probabilities renormalized.

Macro F1 is the headline metric because accuracy rewards collapse. MENG21 posts 0.3715 accuracy on TagaSenti while predicting negative for every row. Its macro F1 is 0.1806. On the human-annotated Taglish set it posts 0.0950 accuracy and 0.0578 macro F1. A model that guesses one class should not look competitive, and here it does not.

Macro F1 averages the F1 score of each class, so a rare class counts as much as a common one. Recall is the share of a true class the model finds. The Mean column is the arithmetic average of the six scoring sets, with each set weighted the same. That choice keeps a large set like TagaSenti from drowning out a smaller set like Lazada.

## What the benchmark does not measure

The overlap audit is not done. `filisenti`'s TagaSenti cell is the known risk. The other cells have not been checked row by row either. Hash the text and compare each model's training data to each test set before treating any cell as clean.

There are no confidence intervals. The caps are 1,001 to 2,000 rows. A gap under roughly two points should be treated as noise, and several gaps in this table are smaller than that.

The new human-annotated set has only 190 neutral rows. The Lazada set has 123. Neutral recall on those sets has a wide error bar. The star-derived neutral label is also noisy, because three-star reviews are not always neutral in meaning.

Four models have inferred label orders. A probe error could move their rows. `raphgonda/FilipinoShopping` is still missing. The notebook has a TensorFlow fallback, but no rows for it appear in the CSV. The optional `letijo03` LoRA adapter was not run, so the model that trained on the Taglish file is absent from the table. That is good for the taglish column's cleanliness, but it means the roster is not exhaustive.

Calibration, latency, model size, and memory use are not measured. A model with a good F1 can still be too slow or too large for a phone. Domain overlap is possible even on the new sets. TagaSenti contains Shopee and Lazada reviews, so the family is domain-adjacent to both new e-commerce sets. The test set revisions are not pinned, so a rerun months from now may load different rows.

TagaSenti labels are partly LLM-generated, according to its model card. The Taglish set is human-annotated, but this repository does not report inter-annotator agreement. Label quality is a limitation on every column.

## Reproduce

```bash
git clone https://github.com/jjjardev/filisenti-benchmark.git
cd filisenti-benchmark
```

Open `filisenti_benchmark_colab.ipynb` in Colab or Jupyter. Run the cells top to bottom. The notebook installs its own dependencies and writes `benchmark_results.csv` and `benchmark_chart.png`. Set `SAMPLE_CAP` to 500 for a smoke test, then put it back to 2,000 for the full run.

The notebook requires `transformers>=4.44,<5` and installs sentencepiece, protobuf, datasets, scikit-learn, accelerate, pandas, and matplotlib. A GPU is optional. The default run fits a free Colab T4.

## Usage

Install the runtime with `pip install transformers torch`. The pipeline returns one label and one score per input.

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="jjjardev/tagasenti_model",
    tokenizer="jjjardev/tagasenti_model",
    truncation=True,
    max_length=128,
)

texts = [
    "Sobrang ganda ng quality ng tela, sulit na sulit ang bayad",
    "Wala pa ring update ang order ko, dalawang linggo na",
]

for row in classifier(texts):
    print(row["label"], round(row["score"], 4))
```

Replace the model ID with `jjjardev/filisenti` or `jjjardev/hilisenti-v1-model` to use the other checkpoints. The three checkpoints expose negative, neutral, and positive labels in their configs. Feed them raw text. HiliSenti performs best on lowercased, normalized Hiligaynon input.

## Repository layout

| Path | Contents |
|---|---|
| `README.md` | This document |
| `LICENSE` | MIT license for the code and files in this repository |
| `benchmark_results.csv` | All 81 scored rows, with accuracy, macro F1, and per-class recall |
| `benchmark_chart.png` | Macro F1 chart for the ten three-class models on the six scoring sets |
| `filisenti_benchmark_colab.ipynb` | Notebook that produces the CSV and chart |

## Citation

```bibtex
@misc{jarder2026filisenti,
  title = {FiliSenti Benchmark},
  author = {Jarder, Jessie James},
  year = {2026},
  url = {https://github.com/jjjardev/filisenti-benchmark}
}
```

## Changelog

- `v2.0.0`: added the human-annotated Taglish set, the Lazada set, and the gadgets contamination probe. Rebuilt the chart and notebook. Added the clean and home-turf framing and the FiliSenti overlap caveat.
- `v1.2.0`: expanded README with the scoreboard, baseline analysis, model guide, metric definitions, and reproduction notes. Corrected the MTEB star mapping.
- `v1.1.0`: renamed the repository to filisenti-benchmark. Added the DOST-ASTI comparison and per-class recall table.
- `v1.0.0`: first public run, chart, notebook, and MIT license.

## License

The code, notebook, and files in this repository use the MIT license. See `LICENSE`. The model weights and datasets keep their own terms, so check the cards before commercial use.

## Links

Models: [filisenti](https://huggingface.co/jjjardev/filisenti), [hilisenti-v1-model](https://huggingface.co/jjjardev/hilisenti-v1-model), [tagasenti_model](https://huggingface.co/jjjardev/tagasenti_model)

Datasets: [TagaSenti](https://huggingface.co/datasets/jjjardev/tagasenti), [HiliSenti](https://huggingface.co/datasets/jjjardev/hilisenti-v1), [letijo03/shopee-sentiment](https://huggingface.co/datasets/letijo03/shopee-sentiment), [magixxixx computer reviews](https://huggingface.co/datasets/magixxixx/shopee-product-reviews-on-computer-category)

Benchmark outputs: [benchmark_results.csv](benchmark_results.csv), [benchmark_chart.png](benchmark_chart.png)

Notebook: [filisenti_benchmark_colab.ipynb](filisenti_benchmark_colab.ipynb)
