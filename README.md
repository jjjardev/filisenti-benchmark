# FiliSenti Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-jjjardev-yellow.svg)](https://huggingface.co/jjjardev)

Zero-shot benchmark for three Filipino sentiment classifiers from jjjardev: `filisenti`, `hilisenti-v1-model`, and `tagasenti_model`. Ten public models were scored on the same test sets where their label space allowed, through the same scoring code. Every score in this README comes from `benchmark_results.csv`.

## Contents

- [Overview](#overview)
- [Results](#results)
- [Per-class recall](#per-class-recall)
- [Against DOST-ASTI](#against-dost-asti)
- [Models](#models)
- [How the run works](#how-the-run-works)
- [Reproduce](#reproduce)
- [Usage](#usage)
- [Intended use](#intended-use)
- [Limitations](#what-the-benchmark-does-not-measure)
- [License and links](#license)

## Overview

`filisenti` posts the highest mean macro F1 among 3-class models at 0.8276. `hilisenti-v1-model` follows at 0.7549 and `tagasenti_model` at 0.7190. The best model outside the family, `cardiffnlp/twitter-xlm-roberta-base-sentiment`, reaches 0.6492. The top three mean scores are jjjardev.

The family does not win every column. HiliSenti takes Hiligaynon at 0.9287, just above FiliSenti at 0.9272. FiliSenti takes TagaSenti at 0.9327. On the Shopee binary set, the best specialist, `scaredmeow/xlnet-tagalog-sentiment-shopee`, scores 0.9480 against 0.8825 for the best jjjardev entry. On MTEB, the three jjjardev models land between 0.5648 and 0.5680, behind `immortalizing-legacies` at 0.5881.

That is the honest shape of the run. The family is strong across the 3-class sets and competitive on binary, yet still short of the best domain specialists on their home turf.

## Scoreboard

| Test set | Winner | Score | Best jjjardev | Score |
|---|---|---:|---|---:|
| TagaSenti | jjjardev/filisenti | 0.9327 | jjjardev/filisenti | 0.9327 |
| Hiligaynon | jjjardev/hilisenti-v1-model | 0.9287 | jjjardev/hilisenti-v1-model | 0.9287 |
| Shopee binary | scaredmeow/xlnet-tagalog-sentiment-shopee | 0.9480 | jjjardev/filisenti | 0.8825 |
| MTEB Shopee | immortalizing-legacies/immortalizing-legacies-sentiment | 0.5881 | jjjardev/filisenti | 0.5680 |
| Mean, 3-class | jjjardev/filisenti | 0.8276 | jjjardev/filisenti | 0.8276 |

Two of the four test sets go to the family. The other two go to specialists with a narrower training mix.

## Why this benchmark exists

Most public Filipino sentiment models are trained on Shopee or Lazada reviews. That is a useful niche. It is also a narrow one. A model tuned on product ratings can learn to read "worth it" and "ang mahal" without learning what a complaint looks like in a news comment or a student evaluation.

This benchmark asks a simpler question. How do the models hold up on three-class sentiment across domains, and how much of the score comes from familiar storefront vocabulary? The answer is mixed, and the mix is informative. The jjjardev family owns the 3-class mean, the public baselines split the specialist columns, and one government model with far more training resources still trails on its own language.

## Results

Macro F1 by test set. The Mean column weights each set equally, which is a choice about this table rather than a claim about production traffic. Binary-only models are excluded from the mean and the table because a model that never outputs neutral cannot be scored fairly on a three-class test.

| Model | TagaSenti | Hiligaynon | Shopee binary | MTEB Shopee | Mean |
|---|---:|---:|---:|---:|---:|
| jjjardev/filisenti | 0.9327 | 0.9272 | 0.8825 | 0.5680 | 0.8276 |
| jjjardev/hilisenti-v1-model | 0.6454 | 0.9287 | 0.8808 | 0.5648 | 0.7549 |
| jjjardev/tagasenti_model | 0.8475 | 0.5850 | 0.8768 | 0.5665 | 0.7190 |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | 0.6152 | 0.5583 | 0.8860 | 0.5374 | 0.6492 |
| immortalizing-legacies/immortalizing-legacies-sentiment | 0.5880 | 0.3925 | 0.9120 | 0.5881 | 0.6202 |
| tabularisai/multilingual-sentiment-analysis | 0.5403 | 0.5044 | 0.8070 | 0.4727 | 0.5811 |
| dost-asti/RoBERTa-tl-sentiment-analysis | 0.4926 | 0.4687 | 0.7827 | 0.4490 | 0.5483 |
| sdbrgo/roberta-tagalog-sentiment-multiclass-classifier | 0.4192 | 0.4614 | 0.6805 | 0.3351 | 0.4741 |
| magixxixx/roberta-shopee-sentiment-gadgets | 0.3872 | 0.2166 | 0.6922 | 0.4117 | 0.4269 |
| MENG21/stud-fac-eval-bert-base-uncased_v2 | 0.1806 | 0.1797 | 0.3333 | 0.1905 | 0.2210 |

![Macro F1 by model and dataset](benchmark_chart.png)

Three binary checkpoints sit outside this table. `scaredmeow/xlnet-tagalog-sentiment-shopee` scores 0.9480 on the Shopee binary set, `The-Thesis-Gods/lazada-xlm-roberta-sentiment` scores 0.8917, and `scaredmeow/roberta-tagalog-sentiment-shopee` scores 0.9090. The last one appears on the 3-class sets in the CSV with zero neutral recall, which is exactly why it is not ranked here.

## Reading the other models

`cardiffnlp/twitter-xlm-roberta-base-sentiment` is the strongest generalist outside the family. It averages 0.6492, with its best 3-class result at 0.6152 on TagaSenti. A multilingual Twitter model should not be the ceiling for Filipino sentiment, and here it is not close to the jjjardev scores.

`immortalizing-legacies/immortalizing-legacies-sentiment` wins MTEB at 0.5881 and posts 0.9120 on the binary set, the second-best binary score in the run. Then Hiligaynon drops to 0.3925. That is a model with a wide English and e-commerce base and little Philippine-language depth.

`tabularisai/multilingual-sentiment-analysis` collapses five classes into three. It averages 0.5811. On Hiligaynon it recalls 0.6991 of neutral rows but only 0.2841 of positive rows, so the collapse pushes it toward the middle. The mapping costs information that the three-class models keep.

`magixxixx/roberta-shopee-sentiment-gadgets` shows the limits of a single storefront. It reaches 0.6922 on Shopee binary. On Hiligaynon it falls to 0.2166, and on TagaSenti it gets 0.3872.

`sdbrgo/roberta-tagalog-sentiment-multiclass-classifier` trained on synthetic Taglish. It averages 0.4741 and leans on negative, with 0.8708 negative recall on TagaSenti and 0.0781 neutral recall. On MTEB its neutral recall is 0.0.

None of these are bad models. They are narrow models measured outside their lane, which is the point of a cross-domain table.

## Per-class recall

Macro F1 can hide a model that wins by guessing one class. These recalls show where each jjjardev model earns its score. Taga and Hili are shorthand for the TagaSenti and Hiligaynon test sets.

| Model | Taga neg | Taga neu | Taga pos | Hili neg | Hili neu | Hili pos | MTEB neu |
|---|---:|---:|---:|---:|---:|---:|---:|
| jjjardev/tagasenti_model | 0.8439 | 0.8515 | 0.8477 | 0.6111 | 0.5805 | 0.5667 | 0.1825 |
| jjjardev/filisenti | 0.9233 | 0.9541 | 0.9205 | 0.9593 | 0.8991 | 0.9211 | 0.1975 |
| jjjardev/hilisenti-v1-model | 0.6070 | 0.6815 | 0.6457 | 0.9363 | 0.9115 | 0.9383 | 0.4050 |

The three TagaSenti recalls sit inside a span of 0.0076. No class carries the score. FiliSenti keeps every Hiligaynon recall above 0.899. HiliSenti holds its home language above 0.911 on all three classes.

The MTEB neutral column is the weak spot. HiliSenti is the best of the three at 0.4050 and still misses about six of every ten neutral reviews. TagaSenti catches 0.1825. The MTEB set builds neutral from three-star reviews alone, and that narrow bucket gives every model trouble.

## Against DOST-ASTI

The comparison to study is `dost-asti/RoBERTa-tl-sentiment-analysis`. DOST-ASTI is a publicly funded research institute. Its model card lists more than 8 million training instances, an Nvidia V100-32GB, and the COARE computing facility. This is not a small operation.

It scores 0.4926 macro F1 on TagaSenti, with accuracy at 0.528. `tagasenti_model` scores 0.8475 on the same 2,000 rows. `filisenti` scores 0.9327.

The neutral class is the clearest split. DOST-ASTI recalls 0.1930 of neutral reviews. `tagasenti_model` recalls 0.8515. Both models saw the same labels and text. A Tagalog model from a government lab missed about four of every five neutral reviews.

| Test set | dost-asti | tagasenti_model | filisenti | hilisenti-v1-model |
|---|---:|---:|---:|---:|
| TagaSenti | 0.4926 | 0.8475 | 0.9327 | 0.6454 |
| Hiligaynon | 0.4687 | 0.5850 | 0.9272 | 0.9287 |
| Shopee binary | 0.7827 | 0.8768 | 0.8825 | 0.8808 |
| MTEB Shopee | 0.4490 | 0.5665 | 0.5680 | 0.5648 |

Every jjjardev model beats DOST-ASTI on every test set. The Tagalog column matters most, since DOST-ASTI trained for Tagalog. The margin is 0.3549 against `tagasenti_model` and 0.4401 against `filisenti`. On Hiligaynon, FiliSenti reaches 0.9272 against 0.4687. The MTEB gap is the smallest, near 0.12, and it does not flip.

Fairness note. DOST-ASTI built its model for general Tagalog sentiment, not for these splits. Four models in this run ship without a usable label order, and DOST-ASTI is one, so the notebook probes it first. A probe mistake could move the row, but not by 0.35. The gap is bigger than the uncertainty.

## Models

| Model | Base | Language focus | Best 3-class result | Use when |
|---|---|---|---|---|
| `jjjardev/filisenti` | XLM-R large | Tagalog and Hiligaynon | 0.9327 TagaSenti | You need one checkpoint for both languages |
| `jjjardev/hilisenti-v1-model` | XLM-R | Hiligaynon | 0.9287 Hiligaynon | You need Hiligaynon only |
| `jjjardev/tagasenti_model` | XLM-R large | Tagalog and Taglish | 0.8475 TagaSenti | You need a Tagalog specialist |

`tagasenti_model` is the Tagalog model. It scores 0.8475 on TagaSenti and 0.8768 on the binary Shopee set. Its Hiligaynon score drops to 0.5850. Its MTEB score drops further, to 0.5665, with neutral recall at 0.1825. The Hiligaynon drop is expected. The model is trained for Tagalog and Taglish, not cross-lingual transfer.

`filisenti` covers both languages and leads the table. It gives up 0.0015 to HiliSenti on Hiligaynon and 0.0201 to the MTEB leader. `hilisenti-v1-model` is the Hiligaynon specialist, with a 0.6454 on TagaSenti when the language changes. All three are on the Hugging Face Hub.

## Choosing a model

| Task | Start with | Why |
|---|---|---|
| Tagalog reviews, comments, or general text | `tagasenti_model` | 0.8475 on TagaSenti with even per-class recall |
| Tagalog and Hiligaynon in one pipeline | `filisenti` | Highest mean and a small cross-language penalty |
| Hiligaynon only | `hilisenti-v1-model` | 0.9287 on Hiligaynon, the best in the run |
| Binary e-commerce sentiment only | `scaredmeow/xlnet-tagalog-sentiment-shopee` | 0.9480 on Shopee binary against 0.8825 for the best jjjardev entry |

That last row points outside the family on purpose. If the task never needs neutral and the text looks like Shopee reviews, the specialist wins.

## How the run works

| Test set | Rows | Labels | Construction |
|---|---:|---|---|
| `tagasenti_test` | 2,000 | negative, neutral, positive | Rebuilt from the TagaSenti CSV with a stratified 80/10/10 split, seed 42 |
| `hilisenti_test` | 2,000 | negative, neutral, positive | Released HiliSenti test split |
| `shopee_binary` | 1,000 | negative, positive | Released binary test split |
| `mteb_shopee_3class` | 2,000 | negative, neutral, positive | MTEB test split, one and two star reviews negative, three star reviews neutral, four and five star reviews positive |

The TagaSenti split is rebuilt from a single CSV. The other three come from released test splits. Each set is capped, 2,000 rows for the 3-class sets and 1,000 for the binary set, to keep the full run inside a free Colab T4 session. The cap uses seed 42 and a stratified draw.

Sources: [TagaSenti](https://huggingface.co/datasets/jjjardev/tagasenti), [HiliSenti](https://huggingface.co/datasets/jjjardev/hilisenti-v1), [scaredmeow/shopee-reviews-tl-binary](https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-binary), and [MTEB FilipinoShopeeReviewsClassification](https://huggingface.co/datasets/mteb/FilipinoShopeeReviewsClassification).

The class counts are 743 negative, 653 neutral, and 604 positive on TagaSenti. Hiligaynon has 738, 565, and 697 in the same order. MTEB has 800, 400, and 800, and the binary set is an even 500 and 500.

Models run at `MAX_LEN=128` with batch size 32 and fp16 autocast on GPU. The notebook reads `id2label` from each config when the labels are named. Four models have no usable label order, so a probe with two sentences per sentiment prints the inferred map before scoring. Two-label models run only on the binary set. Three-class models on the binary set get their negative and positive probabilities renormalized.

Macro F1 is the headline metric because accuracy rewards collapse. MENG21 posts 0.3715 accuracy on TagaSenti while predicting negative for every row. Its macro F1 is 0.1806. The per-class recalls show the same failure at a glance. A model that guesses one class should not look competitive, and here it does not.

Macro F1 averages the F1 score of each class, so a rare class counts as much as a common one. Recall is the share of a true class the model finds. Mean is the arithmetic average of the four macro F1 scores, with each test set weighted the same. Zero-shot means no model was fine-tuned on these test rows in this run. It does not prove a model never saw similar text during pretraining.

## Adding another model

Add the Hub ID to the `MODELS` list in the notebook. Models with named `id2label` entries need nothing else. Models with `LABEL_0`, `LABEL_1`, and `LABEL_2` go through the probe cell, and the printed map needs a check before the score means anything. Two-label models should run on `shopee_binary` only. Set `SAMPLE_CAP` to 500 for a first pass.

## Intended use

This repository is for comparing Filipino sentiment models and for choosing a checkpoint. It is not a moderation system. A sentiment label is not a measure of intent, sarcasm, or safety. A positive review can still be abusive, and a negative review can be fair criticism. Test the model on your own domain before deployment.

## What the benchmark does not measure

Calibration, latency, model size, and memory use are absent. A model with a good F1 can still be too slow or too large for a phone. The caps widen the confidence interval, and no statistical test appears here. Treat small gaps as noise. The 0.0015 between HiliSenti and FiliSenti on Hiligaynon is one of those gaps.

The TagaSenti split is a reconstruction and may not match the published split. Four label orders are inferred. `raphgonda/FilipinoShopping` is missing because the repository ships TensorFlow weights only and the notebook loads PyTorch. MENG21 uses an English BERT base on a Filipino task, and its all-negative collapse makes that row a floor rather than a fair comparison. `tabularisai` collapses five classes into three, which costs it information the other models keep.

Contamination is possible. Several public models trained on Shopee reviews, so the binary column is friendlier to them than the 3-class sets are. The cleanest comparison is the Hiligaynon column for Tagalog-side models, because fewer competitors trained on that language. The DOST-ASTI gap survives every caveat in this section.

## Reproduce

```bash
git clone https://github.com/jjjardev/filisenti-benchmark.git
cd filisenti-benchmark
```

Open `tagasenti_full_benchmark_colab.ipynb` in Colab or Jupyter. Run the cells top to bottom. The notebook installs its own dependencies and writes `benchmark_results.csv` and `benchmark_chart.png`. Set `SAMPLE_CAP` to 500 for a smoke test, then put it back to 2,000 for the full run.

The notebook requires `transformers>=4.44,<5` and installs sentencepiece, protobuf, datasets, scikit-learn, accelerate, pandas, and matplotlib. A GPU is optional. The default run fits a free Colab T4.

## Usage

Install the runtime with `pip install transformers torch`. The pipeline returns one label and one score per input.

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="jjjardev/filisenti",
    tokenizer="jjjardev/filisenti",
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

Replace the model ID with `jjjardev/tagasenti_model` or `jjjardev/hilisenti-v1-model` to use the other checkpoints. The three checkpoints expose negative, neutral, and positive labels in their configs. Feed them raw text. HiliSenti performs best on lowercased, normalized Hiligaynon input.

## Repository layout

| Path | Contents |
|---|---|
| `README.md` | This document |
| `benchmark_results.csv` | All 46 scored rows, with accuracy, macro F1, and per-class recall |
| `benchmark_chart.png` | Macro F1 chart used above |
| `tagasenti_full_benchmark_colab.ipynb` | Notebook that produces the CSV and chart |
| `LICENSE` | MIT license for the code and files in this repository |

## Acknowledgements

The test sets come from TagaSenti, HiliSenti, the scaredmeow Shopee binary split, and the MTEB FilipinoShopeeReviewsClassification task. The public checkpoints belong to their respective authors. The DOST-ASTI model card supplied the training details quoted in the comparison above.

## Citation

```bibtex
@misc{jarder2026filisenti,
  title = {FiliSenti Benchmark},
  author = {Jarder, Jessie James},
  year = {2026},
  url = {https://github.com/jjjardev/filisenti-benchmark}
}
```

## Corrections

If a score or label map looks wrong, open an issue with the row name and the notebook cell that produced it. The CSV is the source of truth, and a correction should start there.

## Changelog

- `v1.2.0`: expanded README with the scoreboard, baseline analysis, model guide, metric definitions, and reproduction notes. Corrected the MTEB star mapping.
- `v1.1.0`: renamed the repository to filisenti-benchmark. Added the DOST-ASTI comparison and per-class recall table.
- `v1.0.0`: first public run, chart, notebook, and MIT license.

## License

The code, notebook, and files in this repository use the MIT license. See `LICENSE`. The model weights and datasets keep their own terms, so check the cards before commercial use.

## Links

Models: [filisenti](https://huggingface.co/jjjardev/filisenti), [hilisenti-v1-model](https://huggingface.co/jjjardev/hilisenti-v1-model), [tagasenti_model](https://huggingface.co/jjjardev/tagasenti_model)

Datasets: [TagaSenti](https://huggingface.co/datasets/jjjardev/tagasenti), [HiliSenti](https://huggingface.co/datasets/jjjardev/hilisenti-v1)

Benchmark outputs: [benchmark_results.csv](benchmark_results.csv), [benchmark_chart.png](benchmark_chart.png)

Notebook: [tagasenti_full_benchmark_colab.ipynb](tagasenti_full_benchmark_colab.ipynb)
