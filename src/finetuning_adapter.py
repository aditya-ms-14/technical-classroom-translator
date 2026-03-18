"""
Fine-Tuning Based Domain Adaptation
Approach 2: Fine-tune Helsinki-NLP models on technical AI/Engineering sentence pairs.

When OPUS data is present (data/opus/opus_{language}.csv or merged_{language}.csv),
it is used for fine-tuning, giving the model 10 000+ real parallel sentences
in addition to the hand-crafted technical glossary pairs.
"""

import pandas as pd
import os
from pathlib import Path

LANGUAGE_CONFIGS = {
    'hindi':   {'model': 'Helsinki-NLP/opus-mt-en-hi',  'code': 'hi'},
    'tamil':   {'model': 'Helsinki-NLP/opus-mt-en-dra', 'code': 'ta'},
    'telugu':  {'model': 'Helsinki-NLP/opus-mt-en-dra', 'code': 'te'},
    'kannada': {'model': 'Helsinki-NLP/opus-mt-en-dra', 'code': 'kn'},
}


# ── data preparation ─────────────────────────────────────────────────────────

def load_opus_pairs(data_dir, language, max_rows=10000):
    """
    Load OPUS sentence pairs from:
      1. merged_{language}.csv   (OPUS + glossary, preferred)
      2. opus_{language}.csv     (OPUS only)
    """
    opus_dir = Path(data_dir) / "opus"
    candidates = [
        opus_dir / "merged_{}.csv".format(language),
        opus_dir / "opus_{}.csv".format(language),
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path).dropna(subset=["en", "target"])
            df = df[df["en"].str.len() > 5]
            pairs = df[["en", "target"]].head(max_rows).to_dict("records")
            print("  Loaded {:,} OPUS pairs from {}".format(len(pairs), path.name))
            return pairs
    return []


def prepare_finetuning_data(terms_csv, sentences_csv, language,
                             data_dir=None, max_opus_rows=10000):
    """
    Build the full training set:
      - Technical glossary term pairs          (always included)
      - Hand-crafted test sentences            (always included)
      - OPUS parallel sentences (if available)
    """
    terms_df     = pd.read_csv(terms_csv)
    sentences_df = pd.read_csv(sentences_csv)
    pairs = []

    # 1. Glossary term pairs
    for _, row in terms_df.iterrows():
        pairs.append({"en": row["term"], "target": row[language]})
        pairs.append({"en": "{}: {}".format(row["term"], row["definition"]),
                      "target": row[language]})

    # 2. Technical sentences
    if language in sentences_df.columns:
        for _, row in sentences_df.iterrows():
            if pd.notna(row.get(language)):
                pairs.append({"en": row["english_sentence"], "target": row[language]})

    # 3. OPUS data (bulk)
    if data_dir:
        opus_pairs = load_opus_pairs(data_dir, language, max_rows=max_opus_rows)
        pairs.extend(opus_pairs)

    # Deduplicate on English side
    seen = set()
    unique = []
    for p in pairs:
        key = p["en"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    print("  Total training pairs for {}: {:,}".format(language, len(unique)))
    return unique


# ── fine-tuning ──────────────────────────────────────────────────────────────

def finetune_model(language, train_pairs, output_dir, num_epochs=3, batch_size=8):
    """Fine-tune the baseline NMT model on domain-specific pairs."""
    try:
        from transformers import (
            MarianMTModel, MarianTokenizer,
            Seq2SeqTrainer, Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq,
        )
        from datasets import Dataset
    except ImportError:
        raise ImportError("Run: pip install transformers datasets torch sentencepiece")

    config     = LANGUAGE_CONFIGS[language]
    model_name = config["model"]
    save_path  = os.path.join(output_dir, "finetuned_{}".format(language))

    print("\nFine-tuning {} for {} ...".format(model_name, language))
    print("  Training samples : {:,}".format(len(train_pairs)))
    print("  Epochs           : {}".format(num_epochs))

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model     = MarianMTModel.from_pretrained(model_name)

    def tokenize(examples):
        model_inputs = tokenizer(
            examples["en"], max_length=128, truncation=True, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"], max_length=128, truncation=True, padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset   = Dataset.from_list(train_pairs)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["en", "target"])

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = save_path,
        num_train_epochs            = num_epochs,
        per_device_train_batch_size = batch_size,
        warmup_steps                = 200,
        weight_decay                = 0.01,
        logging_dir                 = os.path.join(save_path, "logs"),
        logging_steps               = 50,
        save_strategy               = "epoch",
        predict_with_generate       = True,
        fp16                        = False,
        report_to                   = "none",
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = training_args,
        train_dataset = tokenized,
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        tokenizer     = tokenizer,
    )

    trainer.train()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("  Fine-tuned model saved -> {}".format(save_path))
    return save_path


# ── inference ─────────────────────────────────────────────────────────────────

def translate_with_finetuned(texts, model_path, language):
    """Translate using the fine-tuned model."""
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model     = MarianMTModel.from_pretrained(model_path)
    model.eval()

    translations = []
    for text in texts:
        inputs     = tokenizer(text, return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
        translated = model.generate(**inputs, num_beams=4, max_length=512)
        result     = tokenizer.decode(translated[0], skip_special_tokens=True)
        translations.append(result)
    return translations


# ── pipeline runner ───────────────────────────────────────────────────────────

def run_finetuning_pipeline(terms_csv, sentences_csv, baseline_csv,
                             output_dir, languages=None, max_opus_rows=10000):
    """Full fine-tuning pipeline with optional OPUS data."""
    if languages is None:
        languages = list(LANGUAGE_CONFIGS.keys())

    os.makedirs(output_dir, exist_ok=True)
    base_dir     = Path(terms_csv).parent
    sentences_df = pd.read_csv(sentences_csv)
    results      = sentences_df.copy()

    if os.path.exists(baseline_csv):
        baseline_df = pd.read_csv(baseline_csv)
        for lang in languages:
            col = "baseline_{}".format(lang)
            if col in baseline_df.columns:
                results[col] = baseline_df[col]

    for language in languages:
        print("\n" + "="*55)
        print("  {}".format(language.upper()))
        print("="*55)

        train_pairs = prepare_finetuning_data(
            terms_csv     = terms_csv,
            sentences_csv = sentences_csv,
            language      = language,
            data_dir      = base_dir,
            max_opus_rows = max_opus_rows,
        )

        finetuned_path = finetune_model(
            language    = language,
            train_pairs = train_pairs,
            output_dir  = output_dir,
            num_epochs  = 3,
            batch_size  = 4,
        )

        print("  Translating test sentences with fine-tuned model ...")
        finetuned_translations = translate_with_finetuned(
            sentences_df["english_sentence"].tolist(),
            finetuned_path,
            language,
        )
        results["finetuned_{}".format(language)] = finetuned_translations
        print("  {} done".format(language))

    out_path = os.path.join(output_dir, "finetuned_translations.csv")
    results.to_csv(out_path, index=False)
    print("\nFine-tuned translations saved -> {}".format(out_path))
    return results


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    run_finetuning_pipeline(
        terms_csv     = str(base / "data" / "technical_terms.csv"),
        sentences_csv = str(base / "data" / "test_sentences.csv"),
        baseline_csv  = str(base / "results" / "baseline_translations.csv"),
        output_dir    = str(base / "results"),
        languages     = ["hindi", "tamil", "telugu", "kannada"],
        max_opus_rows = 10000,
    )
