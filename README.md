# Improving Domain-Specific Accuracy in Real-Time Classroom Translation

> **Mini Project** based on the case study: *AI-Driven Real-Time Multilingual Communication (Microsoft Presentation Translator)*

---

## Research Question

> *Can domain-specific adaptation improve translation accuracy for technical classroom content?*

---

## Problem Statement

General-purpose Neural Machine Translation (NMT) models (like those powering Microsoft Presentation Translator) frequently **mistranslate technical AI/Engineering vocabulary**. Terms like *backpropagation*, *gradient descent*, and *attention mechanism* either get left in English or translated incorrectly, breaking comprehension for non-English speaking students in technical classrooms.

---

## Approach

This project compares **three translation strategies** on a curated dataset of 25 AI/ML/Engineering sentences:

| # | Method | Description |
|---|--------|-------------|
| 1 | **Baseline NMT** | Helsinki-NLP `opus-mt` models — no domain adaptation |
| 2 | **Glossary Adaptation** | Post-edit baseline output using a 40-term technical glossary |
| 3 | **Fine-Tuning** | Fine-tune baseline model on domain-specific sentence pairs |

---

## Languages

| Language | Script | Model Target Code |
|----------|--------|-------------------|
| Hindi    | Devanagari | `hi` |
| Tamil    | Tamil | `ta` |
| Telugu   | Telugu | `te` |
| Kannada  | Kannada | `kn` |

---

## Project Structure

```
├── data/
│   ├── technical_terms.csv      # 40-term AI/Engineering glossary with translations
│   ├── test_sentences.csv       # 25 domain-specific test sentences
│   └── glossary.json            # Exported glossary (generated)
├── src/
│   ├── baseline_translator.py   # HuggingFace NMT baseline pipeline
│   ├── glossary_adapter.py      # Glossary-based post-editing adapter
│   ├── finetuning_adapter.py    # Fine-tuning domain adaptation
│   └── evaluator.py             # BLEU + Term Accuracy evaluation
├── notebooks/
│   └── analysis.ipynb           # Exploratory analysis & results visualization
├── results/                     # Auto-generated outputs
│   ├── baseline_translations.csv
│   ├── glossary_adapted_translations.csv
│   ├── finetuned_translations.csv
│   ├── evaluation_report.md
│   ├── evaluation_report.json
│   └── comparison_chart.png
├── main.py                      # Pipeline orchestrator
└── requirements.txt
```

---

## Dataset: OPUS Parallel Corpus

This project uses **OPUS** (Open Parallel Corpus) — one of the largest freely available collections of translated texts on the web.

| Source | Description | Size |
|--------|-------------|------|
| **OPUS-100** | Sampled 100-language corpus | ~1M pairs/language |
| **OPUS Books** | Translated literary texts | 100k+ pairs |
| **Tatoeba MT** | Short everyday sentences | 50k+ pairs |
| **WikiMatrix** | Wikipedia-mined parallel sentences | 100k+ pairs |

The downloader pulls up to **10,000 pairs per language** by default (configurable up to millions) and merges them with the hand-crafted technical glossary so domain-specific terms are always correctly represented.

### How to download OPUS data

```bash
# Default: 10,000 pairs per language
python main.py download

# Custom size (e.g. 50,000 pairs)
python main.py download --opus-size 50000
```

Downloaded files are saved to `data/opus/`:
- `opus_{language}.csv` — raw OPUS pairs
- `merged_{language}.csv` — OPUS + technical glossary (used for fine-tuning)

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run demo (no model download)
```bash
python main.py demo
```

### 3. Run full pipeline
```bash
# All steps (baseline + glossary + fine-tune + evaluate)
python main.py all

# Skip fine-tuning (faster, glossary adaptation only)
python main.py all --skip-finetune
```

### 4. Run individual steps
```bash
python main.py baseline    # Step 1: Baseline translations
python main.py glossary    # Step 2: Glossary adaptation
python main.py finetune    # Step 3: Fine-tuning
python main.py evaluate    # Step 4: Evaluation
```

### 5. Jupyter Notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Dataset

### Technical Terms (`data/technical_terms.csv`)
40 AI/ML/Engineering terms with:
- English term + definition
- Translations in Hindi, Tamil, Telugu, Kannada
- Domain tag (AI/ML, NLP, ML)

Sample:

| Term | Domain | Hindi | Tamil | Telugu | Kannada |
|------|--------|-------|-------|--------|---------|
| Neural Network | AI/ML | न्यूरल नेटवर्क | நரம்பியல் வலையமைப்பு | నాడీ నెట్‌వర్క్ | ನರ್ವಲ್ ನೆಟ್‌ವರ್ಕ್ |
| Backpropagation | AI/ML | बैकप्रोपेगेशन | பின்னோக்கி பரப்புதல் | బ్యాక్‌ప్రాపగేషన్ | ಬ್ಯಾಕ್‌ಪ್ರಾಪಗೇಷನ್ |
| Transformer | AI/ML | ट्रांसफार्मर | மாற்றி | ట్రాన్స్ఫార్మర్ | ಟ್ರಾನ್ಸ್‌ಫಾರ್ಮರ್ |

### Test Sentences (`data/test_sentences.csv`)
25 sentences covering:
- Core ML concepts (neural networks, optimization, regularization)
- NLP tasks (tokenization, translation, NER, sentiment analysis)
- Deep learning architectures (CNN, RNN, GAN, Transformer)

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU Score** | Standard MT quality metric (0–1, higher is better) |
| **Term Accuracy** | % of technical terms correctly translated |

---

## Expected Results

| Method | BLEU | Term Accuracy |
|--------|------|---------------|
| Baseline | Low | Low (terms often kept in English) |
| Glossary | Medium | High (direct term replacement) |
| Fine-Tuned | High | High (learned domain patterns) |

---

## Models Used

- **Hindi**: `Helsinki-NLP/opus-mt-en-hi`
- **Tamil/Telugu/Kannada**: `Helsinki-NLP/opus-mt-en-dra` (Dravidian language family)

---

## Hardware Requirements

| Mode | RAM | GPU |
|------|-----|-----|
| Demo only | 2 GB | Not required |
| Baseline translation | 4 GB | Optional |
| Fine-tuning | 8 GB+ | Recommended (CUDA) |

---

## References

- [Helsinki-NLP/opus-mt Models](https://huggingface.co/Helsinki-NLP)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [BLEU Score — Papineni et al. 2002](https://aclanthology.org/P02-1040/)
- [Microsoft Presentation Translator Case Study](https://www.microsoft.com/en-us/translator/)

---

*Project for: AI-Driven Real-Time Multilingual Communication — Case Study Mini Project*
