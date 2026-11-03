# FiliSenti Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-jjjardev-yellow.svg)](https://huggingface.co/jjjardev)

**Author:** [jjjardev](https://github.com/jjjardev)

This repository scores 13 Filipino sentiment classifiers on six test sets and one contamination probe. The run produces 91 rows. Every score below is read from `benchmark_results.csv`; class counts and subset sizes come from the test sets. The chart is generated from the CSV and has two panels, one per metric. The notebook that produced the CSV is `filisenti_benchmark.ipynb`.

The run contains two different metrics. A **three-class score** is macro F1 over negative, neutral, and positive on all rows. A **neutral-free score** is macro F1 over negative and positive after removing rows whose gold label is neutral. Binary models were scored on both kinds of set. Three-class models were scored on all rows of the three-class sets, and on the binary sets after renormalizing negative vs positive; they were not given neutral-free scores on three-class sets. The two scores are not interchangeable, and this README keeps them in separate tables.

## Contents

- [At a glance](#at-a-glance)
- [Protocol](#protocol)
- [Three-class results](#three-class-results)
- [Neutral-free results](#neutral-free-results)
- [Why the two tables are not one leaderboard](#why-the-two-tables-are-not-one-leaderboard)
- [Per-set winners](#per-set-winners)
- [Neutral recall](#neutral-recall)
- [Observations from the scores](#observations-from-the-scores)
- [What is clean and what is home turf](#what-is-clean-and-what-is-home-turf)
- [Against DOST-ASTI](#against-dost-asti)
- [The FiliSenti TagaSenti overlap](#the-filisenti-tagasenti-overlap)
- [Models](#models)
- [Choosing a model](#choosing-a-model)
- [How the run works](#how-the-run-works)
- [Method notes](#method-notes)
- [Claims the data does and does not support](#claims-the-data-does-and-does-not-support)
- [What this does not measure](#what-this-does-not-measure)
- [Reproduce](#reproduce)
- [Usage](#usage)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [Corrections](#corrections)
- [License](#license)
- [Links](#links)

## At a glance

- **13 models, 7 sets, 91 rows.** The notebook's `MODELS` list has 14 entries; `raphgonda/FilipinoShopping` failed to load.
- **Three-class top three:** `jjjardev/filisenti` 0.7572, `jjjardev/hilisenti-v1-model` 0.7024, `jjjardev/tagasenti_model` 0.6873. The best outsider is `immortalizing-legacies/immortalizing-legacies-sentiment` at 0.6206.
- **Neutral-free top two:** `The-Thesis-Gods/lazada-xlm-roberta-sentiment` 0.8364 and `scaredmeow/xlnet-tagalog-sentiment-shopee` 0.7393. These are the only two models with neutral-free scores.
- **Highest score on a test set:** `scaredmeow/xlnet` 0.9480 on Shopee binary.
- **Best neutral recall by set:** 0.9541 on TagaSenti (`filisenti`), 0.9115 on Hiligaynon (`hilisenti-v1-model`), 0.5895 on Taglish (`tagasenti_model`), 0.5528 on Lazada (`hilisenti-v1-model`), 0.5050 on MTEB (`immortalizing-legacies`).
- **Two models cannot produce a three-class score.** MENG21 and `scaredmeow/roberta` have two output logits. MENG21's three-class rows are a harness artifact; `scaredmeow/roberta`'s three-class-set rows use a different protocol from the other binary models. Neither is in the three-class table.

## Protocol

All models were loaded from their published weights. No model was fine-tuned inside this run. That does not make every column zero-shot: `tagasenti_test` and `hilisenti_test` are the training sets of the jjjardev models, and `gadgets_train_probe` is the training set of `magixxixx`. The three-class columns on those sets are in-domain or train-set scores, not zero-shot scores.

| Test set | n | Labels | Class counts (neg / neu / pos) | Neutral-free n | Source | Role |
|---|---:|---|---|---:|---|---|
| `tagasenti_test` | 2,000 | neg, neu, pos | 743 / 653 / 604 | 1,347 | `jjjardev/tagasenti`, rebuilt seed-42 split | Scoring |
| `hilisenti_test` | 2,000 | neg, neu, pos | 738 / 565 / 697 | 1,435 | `jjjardev/hilisenti-v1` test split | Scoring |
| `shopee_binary` | 1,000 | neg, pos | 500 / 0 / 500 | 1,000 | `scaredmeow/shopee-reviews-tl-binary` test split | Scoring |
| `mteb_shopee_3class` | 2,000 | neg, neu, pos | 800 / 400 / 800 | 1,600 | MTEB `FilipinoShopeeReviewsClassification` test split | Scoring |
| `taglish_annotated_3class` | 2,000 | neg, neu, pos | 190 / 190 / 1,620 | 1,810 | `letijo03/shopee-sentiment` annotated CSV | Scoring |
| `lazada_stars_3class` | 1,001 | neg, neu, pos | 380 / 123 / 498 | 878 | `EricEchemane` Lazada reviews, packaged by SEACrowd | Scoring |
| `gadgets_train_probe` | 1,000 | neg, pos | 500 / 0 / 500 | 1,000 | `magixxixx` computer-category training CSV | Contamination probe |

The TagaSenti test set is rebuilt from the single Hub CSV. The other sets come from released files or public repositories. Each scoring set is capped at 2,000 rows, and the binary sets at 1,000, using seed 42 and a stratified draw. The cap keeps the run inside a free Colab T4 session.

Models run at `MAX_LEN=128`, batch size 32, fp16 autocast on GPU. The notebook reads `id2label` when a config provides it. Four models have no usable label order, so a probe with two sentences per sentiment infers the mapping. The DOST-ASTI mapping is also scored under all six label permutations on TagaSenti.

Three-class models on a binary set have their negative and positive probabilities renormalized. Binary models on a three-class set are scored only on negative and positive gold rows; the exception is `scaredmeow/roberta`, described under [Method notes](#method-notes). Binary models on a binary set are scored on all rows.

Macro F1 is the headline metric. It gives each class equal weight, so predicting the majority class does not by itself produce a high score. `sdbrgo` on the Taglish set illustrates the difference: 0.7230 accuracy, 0.4188 macro F1, because its neutral recall is 0.0053.

## Three-class results

Macro F1 over all three classes, on all rows. The Mean column is the unweighted mean of the six scoring sets.

| Model | TagaSenti | Hiligaynon | Shopee binary | MTEB Shopee | Taglish annotated | Lazada stars | Mean | Mean w/o TagaSenti |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jjjardev/filisenti | 0.9327 | 0.9272 | 0.8825 | 0.5680 | 0.5982 | 0.6344 | **0.7572** | 0.7221 |
| jjjardev/hilisenti-v1-model | 0.6454 | 0.9287 | 0.8808 | 0.5648 | 0.5648 | 0.6300 | **0.7024** | 0.7138 |
| jjjardev/tagasenti_model | 0.8475 | 0.5850 | 0.8768 | 0.5665 | 0.6338 | 0.6141 | **0.6873** | 0.6552 |
| immortalizing-legacies/immortalizing-legacies-sentiment | 0.5880 | 0.3925 | 0.9120 | 0.5881 | 0.6210 | 0.6221 | 0.6206 | 0.6271 |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | 0.6152 | 0.5583 | 0.8860 | 0.5374 | 0.5426 | 0.5813 | 0.6201 | 0.6211 |
| tabularisai/multilingual-sentiment-analysis | 0.5403 | 0.5044 | 0.8070 | 0.4727 | 0.4504 | 0.4930 | 0.5446 | 0.5455 |
| dost-asti/RoBERTa-tl-sentiment-analysis | 0.4926 | 0.4687 | 0.7827 | 0.4490 | 0.4816 | 0.5165 | 0.5318 | 0.5397 |
| sdbrgo/roberta-tagalog-sentiment-multiclass-classifier | 0.4192 | 0.4614 | 0.6805 | 0.3351 | 0.4188 | 0.4459 | 0.4602 | 0.4683 |
| magixxixx/roberta-shopee-sentiment-gadgets | 0.3872 | 0.2166 | 0.6922 | 0.4117 | 0.2974 | 0.3754 | 0.3968 | 0.3987 |

Excluded from this table: `scaredmeow/xlnet` and `The-Thesis-Gods/lazada` (binary heads, no neutral output), MENG21 (two outputs, artifact), and `scaredmeow/roberta` (two outputs, different protocol). `gadgets_train_probe` is not included because it is training data for one model.

## Neutral-free results

Macro F1 over negative and positive only. Five columns remove neutral gold rows; Shopee binary is a native two-class set.

| Model | TagaSenti (n=1,347) | Hiligaynon (n=1,435) | Shopee binary (n=1,000) | MTEB Shopee (n=1,600) | Taglish annotated (n=1,810) | Lazada stars (n=878) | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| The-Thesis-Gods/lazada-xlm-roberta-sentiment | 0.7813 | 0.7387 | 0.8917 | 0.8330 | 0.8867 | 0.8867 | **0.8364** |
| scaredmeow/xlnet-tagalog-sentiment-shopee | 0.7064 | 0.4830 | 0.9480 | 0.7862 | 0.6888 | 0.8234 | **0.7393** |

Only these two rows exist. The other 11 models do not have neutral-free scores on the three-class sets.

## Why the two tables are not one leaderboard

The notebook's `OVERALL` block averages every row a model ran, including `gadgets_train_probe`. That produces:

| Rank | Model | Mean over every row it ran |
|---:|---|---:|
| 1 | The-Thesis-Gods/lazada-xlm-roberta-sentiment | 0.8463 |
| 2 | jjjardev/filisenti | 0.7740 |
| 3 | scaredmeow/xlnet-tagalog-sentiment-shopee | 0.7691 |
| 4 | jjjardev/hilisenti-v1-model | 0.7306 |
| 5 | jjjardev/tagasenti_model | 0.7122 |

That table averages different metrics. Thesis's 0.8463 is five neutral-free subset scores, one native binary score, and the training probe. `filisenti`'s 0.7740 is six full three-class scores plus the training probe. The probe is training data for `magixxixx`; including it would give that model a train-set row and the others a test row. Equal row counts do not mean equal tasks.

The data supports these statements:

- `filisenti` has the highest mean on the six full three-class sets, 0.7572.
- Thesis has the higher of the two neutral-free means, 0.8364 against 0.7393.
- No model in the run has both a three-class score and a neutral-free score on the same set.

It does not support "Thesis beats `filisenti`". That comparison would require scoring both models on both tracks.

## Per-set winners

| Test set | Three-class top | Score | Neutral-free top | Score | Neutral-free rows |
|---|---|---:|---|---:|---:|
| TagaSenti | jjjardev/filisenti | 0.9327 | The-Thesis-Gods/lazada | 0.7813 | 1,347 |
| Hiligaynon | jjjardev/hilisenti-v1-model | 0.9287 | The-Thesis-Gods/lazada | 0.7387 | 1,435 |
| Shopee binary | immortalizing-legacies/immortalizing-legacies-sentiment | 0.9120 | scaredmeow/xlnet | 0.9480 | 1,000 |
| MTEB Shopee | immortalizing-legacies/immortalizing-legacies-sentiment | 0.5881 | The-Thesis-Gods/lazada | 0.8330 | 1,600 |
| Taglish annotated | jjjardev/tagasenti_model | 0.6338 | The-Thesis-Gods/lazada | 0.8867 | 1,810 |
| Lazada stars | jjjardev/filisenti | 0.6344 | The-Thesis-Gods/lazada | 0.8867 | 878 |

Among three-class models, the jjjardev models win four columns (TagaSenti, Hiligaynon, Taglish, Lazada) and `immortalizing-legacies` wins two (Shopee binary, MTEB). The neutral-free column has two entrants; Thesis wins five of six and xlnet wins Shopee binary.

The TagaSenti winner is subject to the overlap question in [The FiliSenti TagaSenti overlap](#the-filisenti-tagasenti-overlap). The Lazada neutral-free winner is subject to the provenance note in [What is clean and what is home turf](#what-is-clean-and-what-is-home-turf).

## Neutral recall

Neutral recall is the share of true neutral rows a model finds. Binary models cannot emit neutral and are not in this table.

| Model | TagaSenti | Hiligaynon | Taglish | Lazada | MTEB |
|---|---:|---:|---:|---:|---:|
| jjjardev/filisenti | 0.9541 | 0.8991 | 0.5053 | 0.2683 | 0.1975 |
| jjjardev/hilisenti-v1-model | 0.6815 | 0.9115 | 0.3474 | 0.5528 | 0.4050 |
| jjjardev/tagasenti_model | 0.8515 | 0.5805 | 0.5895 | 0.2276 | 0.1825 |
| immortalizing-legacies/immortalizing-legacies-sentiment | 0.3492 | 0.0885 | 0.4842 | 0.5122 | 0.5050 |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | 0.4609 | 0.5239 | 0.2632 | 0.2114 | 0.1625 |
| tabularisai/multilingual-sentiment-analysis | 0.4579 | 0.6991 | 0.3316 | 0.0976 | 0.1175 |
| dost-asti/RoBERTa-tl-sentiment-analysis | 0.1930 | 0.2248 | 0.3895 | 0.2358 | 0.0575 |
| sdbrgo/roberta-tagalog-sentiment-multiclass-classifier | 0.0781 | 0.1504 | 0.0053 | 0.0325 | 0.0000 |
| magixxixx/roberta-shopee-sentiment-gadgets | 0.7106 | 0.7628 | 0.5368 | 0.4390 | 0.3575 |

The maxima are: TagaSenti 0.9541 (`filisenti`), Hiligaynon 0.9115 (`hilisenti-v1-model`), Taglish 0.5895 (`tagasenti_model`), Lazada 0.5528 (`hilisenti-v1-model`), MTEB 0.5050 (`immortalizing-legacies`). No model has the maximum on more than two columns.

## Observations from the scores

These are direct comparisons from the tables, not causal claims.

1. **The three-class top three are separated by 0.0699 from first to third.** `filisenti` 0.7572, `hilisenti-v1-model` 0.7024, `tagasenti_model` 0.6873. The third-place jjjardev model is 0.0667 above the best outsider, `immortalizing-legacies` at 0.6206.
2. **No single three-class model wins more than two sets.** `filisenti` wins TagaSenti and Lazada; `hilisenti-v1-model` wins Hiligaynon; `tagasenti_model` wins Taglish; `immortalizing-legacies` wins Shopee binary and MTEB. The family's four columns are split across three checkpoints.
3. **The binary scores span a wide range by set.** Thesis ranges from 0.7387 (Hiligaynon) to 0.8917 (Shopee binary). xlnet ranges from 0.4830 (Hiligaynon) to 0.9480 (Shopee binary). xlnet's gap between its best and worst set is 0.4650; Thesis's is 0.1530.
4. **Hiligaynon scores separate the models by a wide margin.** On the neutral-free Hiligaynon subset, xlnet scores 0.4830 and Thesis 0.7387. On the three-class Hiligaynon set, the jjjardev models score 0.9272 and 0.9287, and the best public three-class model is `cardiffnlp` at 0.5583. The three-class and neutral-free numbers are not comparable. The three-class result is the one where the two models whose cards list HiliSenti are ahead.
5. **The gadgets probe moves one model far more than the others.** `magixxixx` scores 0.9609 on the probe and 0.6922 on Shopee binary, a 0.2687 gap. `filisenti` scores 0.8747 on the probe and 0.8825 on Shopee binary, a -0.0078 difference. `xlnet` scores 0.9480 on both.
6. **Neutral recall is lower on external sets than on the family training sets.** The best values are 0.5895 (Taglish), 0.5528 (Lazada), and 0.5050 (MTEB). On the two family training sets, the best values are 0.9541 and 0.9115. No model in the run is above 0.80 neutral recall on a family training set and above 0.60 on any external set.
7. **The roster is not exhaustive.** `raphgonda/FilipinoShopping` failed to load. The `letijo03` LoRA adapter was not run. Two models with two outputs cannot be scored on the three-class task. The 13 models are the ones that loaded and had a compatible label space, not every Filipino sentiment model on the Hub.

## What is clean and what is home turf

| Test set | Reading | Caveat |
|---|---|---|
| TagaSenti | `tagasenti_model` 0.8475; `hilisenti-v1-model` 0.6454 | `filisenti` 0.9327: its card lists 35,686 TagaSenti rows, which is the entire Hub file, and says the corpus was re-split 80/10/10. This repository does not include a row-level overlap check. |
| Hiligaynon | `filisenti` 0.9272; `hilisenti-v1-model` 0.9287 | Family training set. No public model in the run documents Hiligaynon training. |
| Shopee binary | `xlnet` 0.9480 (binary); `immortalizing-legacies` 0.9120 (three-class) | `xlnet`'s card lists the Shopee binary dataset. `magixxixx`'s probe is Shopee-derived. Several entries are Shopee-trained or Shopee-named. |
| MTEB Shopee | `immortalizing-legacies` 0.5881 (three-class); Thesis 0.8330 (neutral-free) | The neutral label is derived from 3-star ratings; a 3-star review is not necessarily neutral. The neutral-free 0.8330 is a different metric. |
| Taglish annotated | `tagasenti_model` 0.6338 | 190 neutral rows. TagaSenti includes e-commerce reviews, so the domains overlap. Thesis 0.8867 is neutral-free. The `letijo03` adapter trained on this file was not run. |
| Lazada stars | `filisenti` 0.6344 (three-class); Thesis 0.8867 (neutral-free) | The repository name contains "lazada"; the model card does not document training data. Neutral is star-derived. Neutral-free n=878. |
| Gadgets probe | Not a scoring set | Training data for `magixxixx`. Its 0.9609 is a train-set score; every other model sees the rows cold. |

## Against DOST-ASTI

DOST-ASTI's model card describes an Nvidia V100-32GB run on the COARE facility, 5,159,917 formal and 3,057,180 informal training instances, and a published paper (iTANONG-DS). The training data is not TagaSenti or the other test files here.

| Test set | dost-asti | tagasenti_model | filisenti | hilisenti-v1-model |
|---|---:|---:|---:|---:|
| TagaSenti | 0.4926 | 0.8475 | 0.9327 | 0.6454 |
| Hiligaynon | 0.4687 | 0.5850 | 0.9272 | 0.9287 |
| Shopee binary | 0.7827 | 0.8768 | 0.8825 | 0.8808 |
| MTEB Shopee | 0.4490 | 0.5665 | 0.5680 | 0.5648 |
| Taglish annotated | 0.4816 | 0.6338 | 0.5982 | 0.5648 |
| Lazada stars | 0.5165 | 0.6141 | 0.6344 | 0.6300 |

DOST-ASTI is below all four comparison models on all six sets. The Taglish and Lazada columns are the cleanest comparisons, because none of the four models lists those files as training data: `tagasenti_model` leads DOST-ASTI by 0.1522 on Taglish, and `filisenti` leads by 0.1179 on Lazada. Domain overlap with the family's e-commerce training data is still possible.

The DOST-ASTI label order is inferred. On TagaSenti, the six-permutation check gives a best score of 0.4926; the other five permutations range from 0.2004 to 0.3426. The TagaSenti comparison therefore does not depend on one guessed order. The other columns use the best inferred order only.

This run supports "DOST-ASTI scores below these four models on these six sets." It does not support "DOST-ASTI is a worse model in general." A fair test of that claim would need a set in DOST-ASTI's training domain; none is in this repository.

## The FiliSenti TagaSenti overlap

`filisenti` has the highest three-class mean, 0.7572. Its TagaSenti score is 0.9327, and the size of its overlap with the benchmark's TagaSenti test is unknown.

The model card lists 35,686 TagaSenti rows and says the combined corpus was re-split 80/10/10. The Hub dataset has one split with 35,686 rows; the card's "train split only" therefore covers the whole file. The benchmark builds a different 80/10/10 split from the same 35,686 rows with seed 42. This repository does not include a row-level hash check, so the number of benchmark test rows that were also in `filisenti`'s training pool is not reported here.

| Slice | filisenti | hilisenti-v1-model | tagasenti_model |
|---|---:|---:|---:|
| Six scoring sets | 0.7572 | 0.7024 | 0.6873 |
| Five sets, no TagaSenti | 0.7221 | 0.7138 | 0.6552 |

The family is top three on both slices. The `filisenti` lead over `hilisenti-v1-model` falls from 0.0548 on six sets to 0.0083 on five. The TagaSenti score is also higher than the 88.1 Tagalog F1 on `filisenti`'s own card. Neither fact establishes overlap. The row-level check is the missing evidence.

## Models

| Model | Base | Language focus | Best three-class result | Use when |
|---|---|---|---|---|
| `jjjardev/filisenti` | XLM-R large | Tagalog and Hiligaynon | 0.9327 TagaSenti, 0.9272 Hiligaynon | You need one checkpoint for both languages |
| `jjjardev/hilisenti-v1-model` | XLM-R | Hiligaynon | 0.9287 Hiligaynon | You need Hiligaynon only |
| `jjjardev/tagasenti_model` | XLM-R large | Tagalog and Taglish | 0.8475 TagaSenti, 0.6338 Taglish annotated | You need a Tagalog specialist |

`tagasenti_model` scores 0.8475 on TagaSenti, 0.6338 on human-annotated Taglish, and 0.8768 on Shopee binary. Its Hiligaynon score is 0.5850. Its MTEB score is 0.5665, with neutral recall 0.1825.

`filisenti` is 0.0015 below `hilisenti-v1-model` on Hiligaynon and 0.0201 below the MTEB leader. `hilisenti-v1-model` is 0.6454 on TagaSenti.

## Choosing a model

| Task | Start with | Basis |
|---|---|---|
| Tagalog reviews, comments, or general text | `tagasenti_model` | 0.8475 TagaSenti, 0.6338 human-annotated Taglish |
| Tagalog and Hiligaynon in one pipeline | `filisenti` | Highest three-class mean, with the TagaSenti overlap caveat |
| Hiligaynon only | `hilisenti-v1-model` | 0.9287 Hiligaynon |
| Binary Shopee sentiment, no neutral needed | `scaredmeow/xlnet-tagalog-sentiment-shopee` | 0.9480 Shopee binary |
| Binary e-commerce, Lazada-named | `The-Thesis-Gods/lazada-xlm-roberta-sentiment` | 0.8364 neutral-free mean; training data undocumented |
| Three classes, cross-domain, with neutral | No model leads all external sets | Best external neutral recall is 0.5895 |

The binary recommendations apply only when the task has no neutral class. On a three-class task, a binary model cannot emit neutral, and its macro F1 is not comparable to the three-class table.

## How the run works

- 13 models: 3 jjjardev checkpoints and 10 public models.
- 7 sets: 6 scoring sets and 1 contamination probe.
- 91 rows: 13 models × 7 sets.
- Each scoring set is capped at 2,000 rows; binary sets at 1,000. Seed 42, stratified draw.
- `MAX_LEN=128`, batch 32, fp16 autocast on GPU.
- Label order is read from `id2label` when available. Four models use an inferred order: `dost-asti`, `sdbrgo`, MENG21, and `scaredmeow/roberta`.
- DOST-ASTI is additionally scored under all six permutations on TagaSenti.
- Three-class models on Shopee binary renormalize negative and positive.
- Binary models on three-class sets drop neutral gold rows and score negative vs positive; the exception is `scaredmeow/roberta` (see Method notes).
- The gadgets probe is reported but never averaged into a leaderboard.

## Method notes

1. **MENG21's three-class rows are an artifact.** MENG21 has two output logits. Its config gives no `id2label`, and its dataset is not publicly accessible. The probe mapped its outputs to negative and neutral; the three-class fold zeroed the positive column. Every prediction became negative. The 0.1806 TagaSenti row is produced by that fold, not by the model's accuracy on a three-class task.
2. **`scaredmeow/roberta` uses a third protocol.** It also has two output logits. Its probe map is negative/positive, so the positive column survives. On the three-class sets it is scored on all rows with neutral as a class it cannot output: neutral recall 0.0000 and `n` 2,000. That is not the neutral-free protocol used for xlnet and Thesis. Its numbers on Shopee binary (0.9090) and the probe (0.8780) are valid binary scores.
3. **`raphgonda/FilipinoShopping` did not run.** The notebook registers it and sets `from_tf=True, low_cpu_mem_usage=False`, with a comment saying this fixes the meta-tensor error. The recorded output still ends in `NotImplementedError: Cannot copy out of meta tensor`. The CSV has 13 models, not 14.
4. **Nine models have a real neutral output.** MENG21 and `scaredmeow/roberta` have two outputs each, so they are not in the three-class table. `scaredmeow/xlnet` and `The-Thesis-Gods/lazada` are binary and appear only in the neutral-free table.

## Claims the data does and does not support

| Claim | Status | Basis |
|---|---|---|
| `filisenti` has the highest three-class six-set mean. | Supported | 0.7572 vs 0.7024 and 0.6873. |
| The jjjardev models hold the top three three-class means. | Supported | 0.7572, 0.7024, 0.6873 vs best outsider 0.6206. |
| `filisenti`'s TagaSenti 0.9327 is a clean held-out score. | Not established | Its card lists 35,686 TagaSenti rows (the entire Hub file); this repository does not include a row-level overlap check. |
| Thesis is the best model in the run. | Not supported | Its 0.8364 is neutral-free on five subsets; it has no three-class score. |
| Thesis is the best neutral-free model. | Supported among two entrants | 0.8364 vs xlnet 0.7393. No other model has a neutral-free score. |
| `scaredmeow/xlnet` is the best binary Shopee model. | Supported | 0.9480, the highest test score in the run. |
| The jjjardev models are ahead on Hiligaynon. | Supported on the three-class metric | 0.9272 and 0.9287 vs best public 0.5583. |
| Binary models are better than three-class models. | Not supported | Different label spaces; no model has both scores. |
| MENG21 is a weak model. | Not supported | Its rows are harness artifacts. |
| DOST-ASTI is worse in general. | Not supported | It trails on these six sets only; no in-domain set is included. |
| The family generalizes with no loss outside its training sets. | Not supported | Neutral recall falls to 0.1825–0.5895 on external sets. |

## What this does not measure

- **Overlap.** This repository does not include a hash-based training/test overlap check for any model. The `filisenti` TagaSenti cell is the known risk.
- **Confidence intervals.** The sets are 878 to 2,000 rows. Gaps under roughly two points should be treated as noise. The 190 Taglish neutral rows and 123 Lazada neutral rows have wide error bars.
- **Label quality.** The neutral label in MTEB and Lazada comes from 3-star ratings; a 3-star review is not necessarily neutral. TagaSenti is partly LLM-annotated. The Taglish set is human-annotated, but no inter-annotator agreement is reported here.
- **Inferred label orders.** Four models use a probe-inferred order. DOST-ASTI has a six-permutation check on TagaSenti; the other models and columns do not.
- **Missing models.** `raphgonda/FilipinoShopping` failed to load. The `letijo03` LoRA adapter was not run. The roster is not every Filipino sentiment model on the Hub.
- **Paired neutral-free scores.** No three-class model was scored with neutral rows removed. The cross-track comparison is missing by construction.
- **Deployment metrics.** Calibration, latency, model size, and memory use are not measured.
- **Revision pinning.** Test set revisions are not pinned. A rerun months from now may load different rows.
- **Label-space validity for two-output models.** MENG21 needs a two-class path before its rows can be used. `scaredmeow/roberta` needs a neutral-free pass before its three-class-set rows can be compared with the neutral-free table.

## Reproduce

```bash
git clone https://github.com/jjjardev/filisenti-benchmark.git
cd filisenti-benchmark
```

Open `filisenti_benchmark.ipynb` in Colab or Jupyter and run the cells top to bottom. The notebook installs its own dependencies and writes `benchmark_results.csv` and its own `benchmark_chart.png`. Set `SAMPLE_CAP` to 500 for a smoke test, then back to 2,000 for the full run.

The notebook's chart cell plots every row together, including the probe. To rebuild the two-panel chart used here:

```bash
pip install pandas matplotlib
python3 make_chart.py
```

The notebook requires `transformers>=4.44,<5` plus sentencepiece, protobuf, datasets, scikit-learn, accelerate, pandas, and matplotlib. A GPU is optional. The default run fits a free Colab T4.

A rerun will reproduce the MENG21 and `scaredmeow/roberta` rows unless the two-output handling is changed. The CSV is the record of this run, not a claim that every row is valid.

## Usage

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

Replace the model ID with `jjjardev/filisenti` or `jjjardev/hilisenti-v1-model` for the other checkpoints. The three checkpoints expose negative, neutral, and positive labels in their configs. Feed them raw text.

## Repository layout

| Path | Contents |
|---|---|
| `README.md` | This document |
| `LICENSE` | MIT license |
| `benchmark_results.csv` | 91 rows with accuracy, macro F1, and per-class recall |
| `benchmark_chart.png` | Two-panel chart generated by `make_chart.py` |
| `filisenti_benchmark.ipynb` | Notebook that produced the CSV |
| `make_chart.py` | Rebuilds the two-panel chart from the CSV |

## Citation

```bibtex
@misc{jjjardev2026filisenti,
  title = {FiliSenti Benchmark},
  author = {jjjardev},
  year = {2026},
  url = {https://github.com/jjjardev/filisenti-benchmark}
}
```

## Corrections

Open an issue with the row name and the notebook cell that produced it. The CSV is the source of truth. A corrected row should be followed by rerunning `make_chart.py` so the chart matches the CSV.

## License

The code, notebook, and files in this repository use the MIT license. See `LICENSE`. Model weights and datasets keep their own terms; check the cards before commercial use.

## Links

Models: [filisenti](https://huggingface.co/jjjardev/filisenti), [hilisenti-v1-model](https://huggingface.co/jjjardev/hilisenti-v1-model), [tagasenti_model](https://huggingface.co/jjjardev/tagasenti_model)

Test data: [TagaSenti](https://huggingface.co/datasets/jjjardev/tagasenti), [HiliSenti](https://huggingface.co/datasets/jjjardev/hilisenti-v1), [Shopee binary](https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-binary), [MTEB FilipinoShopeeReviewsClassification](https://huggingface.co/datasets/mteb/FilipinoShopeeReviewsClassification), [letijo03/shopee-sentiment](https://huggingface.co/datasets/letijo03/shopee-sentiment), [EricEchemane Lazada reviews](https://github.com/EricEchemane/Filipino-Tagalog-Product-Reviews-Sentiment-Analysis), [magixxixx computer reviews](https://huggingface.co/datasets/magixxixx/shopee-product-reviews-on-computer-category)

Benchmark outputs: [benchmark_results.csv](benchmark_results.csv), [benchmark_chart.png](benchmark_chart.png)

Notebook: [filisenti_benchmark.ipynb](filisenti_benchmark.ipynb)
