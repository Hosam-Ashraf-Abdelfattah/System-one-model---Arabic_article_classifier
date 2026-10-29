---
language: ar
license: apache-2.0
library_name: pytorch
pipeline_tag: text-classification
tags:
  - arabic
  - text-classification
  - topic-classification
  - transformer-encoder
  - calibrated-confidence
  - system-one
  - pytorch
  - custom-code
datasets:
  - arbml/SANAD
metrics:
  - accuracy
  - f1
---

# Arabic SANAD Encoder Classifier

> **Draft release note:** Apache-2.0 is the proposed license for the model code and weights. Confirm this choice before publishing. SANAD dataset attribution and its CC BY 4.0 license must remain in this card.

A compact, non-generative Arabic text classifier built from scratch with PyTorch. It uses a custom BPE tokenizer and a bidirectional Transformer encoder to classify Arabic news into seven topics, returning calibrated confidence and optionally abstaining when confidence is low.

This is a **System-One-style classifier**: it reads text and returns a structured decision rather than generating prose. It is **not** a reproduction of Jev or Laya, does not answer arbitrary user-written questions, and does not yet implement separate yes/no or numeric-score heads.

## Model summary

| Property | Value |
|---|---:|
| Model type | Encoder-only Transformer for sequence classification |
| Parameters | 7,253,255 (~7.25M) |
| Language | Arabic, primarily Modern Standard Arabic news |
| Tasks | Single-label Arabic news topic classification |
| Classes | Tech, Finance, Politics, Religion, Medical, Culture, Sports |
| Tokenizer | Custom word-level pre-tokenization + BPE, 16,000 tokens |
| Maximum input length | 256 tokens, including `[CLS]` and `[SEP]` |
| Hidden size | 256 |
| Encoder layers | 4 |
| Attention heads | 4 |
| Feed-forward size | 1,024 |
| Activation | GELU |
| Normalization | Pre-LayerNorm with a final LayerNorm |
| Positional encoding | Sinusoidal |
| Pooling | Masked mean pooling over real tokens |
| Classification head | Linear(256 → 7), softmax for inference |
| Confidence calibration | Temperature scaling, T = 1.249 |
| Abstention threshold in demo | 0.70 |

## Final evaluation

The model, tokenizer, maximum length, epoch-2 checkpoint, and calibration temperature were frozen before the final test run. The test set contains 14,181 articles.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 95.32% | **95.47%** |
| Macro-F1 | 94.66% | **94.71%** |
| ECE before calibration | 1.84% | **1.53%** |
| ECE after calibration | 0.43% in-sample; about 0.5% in the two-half check | **0.74%** |
| Mean confidence before calibration | 97.17% | **96.99%** |
| Mean confidence after calibration | 95.74% | **95.54%** |
| NLL after calibration | 0.1453–0.1513 in the two-half check | **0.1395** |

Temperature scaling changes confidence values, not class predictions or accuracy.

## Intended use

- Learning how a non-generative encoder Transformer is built and trained from scratch.
- Arabic news topic classification into the seven SANAD categories.
- Experiments with calibrated confidence, selective prediction, and abstention.
- A compact baseline for further Arabic NLP research and education.

## Out-of-scope use

- Text generation, conversation, question answering, or arbitrary decision questions.
- Yes/no (`noul`) or numeric-score (`score`) decisions without additional heads and training.
- High-stakes automated decisions.
- Assuming the reported accuracy transfers to other websites, dialects, time periods, or label taxonomies.

## Quick test

After downloading the complete repository, install its dependencies and run the following from the repository root. The final repository must preserve the paths `data/bpe.json`, `checkpoints/best.pt`, and `checkpoints/temperature.json` expected by the current inference code.

```python
from answer import answer

text = "فاز الفريق بهدفين مقابل هدف في المباراة النهائية أمام جماهيره"
result = answer(text, question="ما موضوع هذا النص؟")
print(result)
```

The returned dictionary has this structure:

```python
{
    "question": "ما موضوع هذا النص؟",
    "type": "choice",
    "answer": "Sports",       # Or None when calibrated confidence is below 0.70
    "confidence": 0.0,        # Actual value is produced by the model
    "top3": []                # Top three classes and their calibrated probabilities
}
```

A second example for manual testing:

```python
text = "أعلنت الشركة ارتفاع أرباحها السنوية وتوزيع أسهم مجانية على المساهمين"
print(answer(text))
```

These examples are demonstrations, not evaluation results. Do not infer accuracy from them.

## Training data

The model was trained on the `arbml/SANAD` Hugging Face version used in this project, containing 141,807 articles. SANAD was originally collected from AlKhaleej, AlArabiya, and Akhbarona for single-label Arabic news categorization.

Project split, stratified by class with seed 42:

| Split | Articles |
|---|---:|
| Train | 113,445 |
| Validation | 14,181 |
| Test | 14,181 |

The classes are imbalanced: Finance is the largest training class (31,485 articles), while Culture is the smallest (5,397). Macro-F1 is therefore reported alongside accuracy.

### SANAD attribution and license

Einea, O., Elnagar, A., & Al Debsi, R. (2019). **SANAD: Single-label Arabic News Articles Dataset for automatic text categorization.** Data in Brief.

- Dataset: https://data.mendeley.com/datasets/57zpx667y9/2
- Paper: https://www.sciencedirect.com/science/article/pii/S2352340919304305
- Dataset license: **CC BY 4.0**. Follow the dataset attribution requirements when redistributing or describing derived work.

## Preprocessing and tokenizer

The custom pipeline performs Arabic normalization, pre-tokenization, BPE encoding, and conversion to token IDs. Normalization removes diacritics and tatweel, unifies several alef forms, converts alef maqsura to yaa, normalizes Arabic-Indic digits, lowercases Latin letters, and collapses whitespace.

The normalization is intentionally simple and lossy. For example, distinctions between “على” and the name “علي” may be removed. This is acceptable for the demonstrated topic-classification task but may be unsuitable for tasks requiring finer Arabic morphology.

A 3,000-article validation sample measured 1.26 tokens per pre-tokenized word. At a 256-token limit, approximately 46% of sampled articles fit completely and approximately 58% of tokens were retained. The model reads the beginning of each article; later text is truncated.

## Training configuration

- Framework: PyTorch, FP32
- Optimizer: AdamW
- Batch size: 32
- Epochs run: 4; best checkpoint selected from epoch 2 by validation loss
- Peak learning rate: 3e-4
- Final learning-rate floor: 3e-5
- Schedule: linear warmup followed by cosine decay
- Weight decay: 0.01 for matrix parameters, 0 for biases and LayerNorm parameters
- Gradient clipping: global norm 1.0
- Loss: cross-entropy over raw logits
- Calibration: temperature fitted on validation logits only, after training

Validation loss was best at epoch 2 (0.1565). Later epochs reduced training loss while validation loss increased, indicating overfitting; the epoch-2 checkpoint is therefore the released model.

## Confidence and abstention

The inference demo divides logits by the validation-fitted temperature **T = 1.249** before softmax. It abstains below calibrated confidence 0.70. On validation, this threshold answered approximately 95.98% of cases with 97.05% accuracy among answered cases.

Calibration does not make the classifier more accurate and does not remove confident errors. It only makes the reported probabilities better aligned with observed accuracy on similar data.

## Limitations

1. **Fixed task:** The model answers one fixed seven-way topic question. It is not a general Jev/Laya-style decision model.
2. **Source overlap:** Train, validation, and test articles come from the same dataset and sources. Performance on a new publisher is unknown.
3. **Label noise and ambiguity:** Manual inspection of a small, non-random error sample found ambiguous or questionable SANAD labels. Reported accuracy is agreement with SANAD labels, not a guaranteed measure of true topic understanding.
4. **Truncation:** Only the first 256 tokens are used, so relevant information near the end of long articles can be lost.
5. **Domain and dialect:** Training is dominated by Modern Standard Arabic news, not general user-generated Arabic or dialectal text.
6. **Calibration limits:** A single global temperature does not calibrate every confidence range or class equally and assumes future data resembles validation data.
7. **No external robustness evaluation:** Adversarial text, temporal shifts, and cross-domain robustness were not measured.

## Reproducibility

The project includes custom implementations of normalization, pre-tokenization, BPE training and encoding, embeddings, sinusoidal positions, multi-head self-attention, feed-forward layers, residual connections, Pre-LayerNorm encoder blocks, masked mean pooling, classification, training, evaluation, temperature calibration, and inference.

The published repository should include source code, model weights, tokenizer data, calibration data, configuration, requirements, and evaluation outputs sufficient to reproduce inference and verify the reported metrics. Raw SANAD articles and large cached training arrays are not required in the model repository.

## Citation

If you use this model, please cite both this repository and the original SANAD dataset/paper. Replace the placeholder before publishing:

```bibtex
@misc{arabic_sanad_encoder_2026,
  author = {Hosam Ashraf},
  title = {Arabic SANAD Encoder Classifier: A From-Scratch Non-Generative Transformer with Calibrated Confidence},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/hosam010023/Arabic_article_classifier}}
}
```
