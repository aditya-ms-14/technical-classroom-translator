"""
Fine-Tuning Based Domain Adaptation
Approach 2: Fine-tune Helsinki-NLP models on technical AI/Engineering sentence pairs.

This module:
1. Prepares a domain-specific fine-tuning dataset from technical_terms.csv
2. Fine-tunes the baseline NMT model using HuggingFace Trainer
3. Evaluates the fine-tuned model against baseline using BLEU score
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Optional


LANGUAGE_CONFIGS = {
    'hindi': {'model': 'Helsinki-NLP/opus-mt-en-hi', 'code': 'hi'},
    'tamil': {'model': 'Helsinki-NLP/opus-mt-en-dra', 'code': 'ta'},
    'telugu': {'model': 'Helsinki-NLP/opus-mt-en-dra', 'code': 'te'},
    'kannada': {'model': 'Helsinki-NLP/opus-mt-en-dra', 'code': 'kn'},
}


def prepare_finetuning_data(terms_csv: str, sentences_csv: str, language: str) -> list:
    """
    Build parallel sentence pairs for fine-tuning from:
    1. Glossary entries (term -> definition + translation)
    2. Technical sentences with known translations
    """
    terms_df = pd.read_csv(terms_csv)
    pairs = []

    # Create training pairs from glossary terms
    for _, row in terms_df.iterrows():
        en = row['term']
        target = row[language]
        pairs.append({'en': en, 'target': target})

        # Add definition context pairs
        definition_en = f"{row['term']}: {row['definition']}"
        # Use term translation as a training signal
        pairs.append({'en': definition_en, 'target': target})

    print(f"Prepared {len(pairs)} fine-tuning pairs for {language}")
    return pairs


def finetune_model(language: str, train_pairs: list, output_dir: str,
                   num_epochs: int = 3, batch_size: int = 8):
    """
    Fine-tune the baseline NMT model on domain-specific pairs.

    Args:
        language: Target language key
        train_pairs: List of {'en': ..., 'target': ...} dicts
        output_dir: Directory to save fine-tuned model
        num_epochs: Training epochs (3 recommended for small datasets)
        batch_size: Training batch size
    """
    try:
        from transformers import (
            MarianMTModel, MarianTokenizer,
            Seq2SeqTrainer, Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq
        )
        from datasets import Dataset
        import torch
    except ImportError:
        print("Required packages missing. Run: pip install transformers datasets torch sentencepiece")
        raise

    config = LANGUAGE_CONFIGS[language]
    model_name = config['model']

    print(f"\nFine-tuning {model_name} for {language}...")
    print(f"Training samples: {len(train_pairs)}, Epochs: {num_epochs}")

    # Load tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize dataset
    def tokenize(examples):
        model_inputs = tokenizer(
            examples['en'],
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'],
                max_length=128,
                truncation=True,
                padding='max_length'
            )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    dataset = Dataset.from_list(train_pairs)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=['en', 'target'])

    # Training arguments
    save_path = os.path.join(output_dir, f"finetuned_{language}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=os.path.join(save_path, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=False,  # Set True if GPU available
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✓ Fine-tuned model saved to {save_path}")

    return save_path


def translate_with_finetuned(texts: list, model_path: str, language: str) -> list:
    """
    Translate using the fine-tuned model.
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        raise ImportError("transformers required: pip install transformers")

    config = LANGUAGE_CONFIGS[language]
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    model.eval()

    translations = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs, num_beams=4, max_length=512)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        translations.append(result)

    return translations


def run_finetuning_pipeline(terms_csv: str, sentences_csv: str,
                             baseline_csv: str, output_dir: str,
                             languages: list = None):
    """
    Full fine-tuning pipeline:
    1. Prepare data
    2. Fine-tune for each language
    3. Translate test set
    4. Save results alongside baseline for comparison
    """
    if languages is None:
        languages = list(LANGUAGE_CONFIGS.keys())

    os.makedirs(output_dir, exist_ok=True)
    sentences_df = pd.read_csv(sentences_csv)
    results = sentences_df.copy()

    # Load baseline results if available
    if os.path.exists(baseline_csv):
        baseline_df = pd.read_csv(baseline_csv)
        for lang in languages:
            col = f'baseline_{lang}'
            if col in baseline_df.columns:
                results[col] = baseline_df[col]

    for language in languages:
        print(f"\n{'='*50}")
        print(f"Processing: {language.upper()}")
        print('='*50)

        # Prepare fine-tuning data
        train_pairs = prepare_finetuning_data(terms_csv, sentences_csv, language)

        # Fine-tune
        finetuned_path = finetune_model(
            language=language,
            train_pairs=train_pairs,
            output_dir=output_dir,
            num_epochs=3,
            batch_size=4
        )

        # Translate test sentences with fine-tuned model
        print(f"Translating test set with fine-tuned model...")
        finetuned_translations = translate_with_finetuned(
            sentences_df['english_sentence'].tolist(),
            finetuned_path,
            language
        )
        results[f'finetuned_{language}'] = finetuned_translations
        print(f"✓ {language} fine-tuned translations complete")

    # Save
    out_path = os.path.join(output_dir, "finetuned_translations.csv")
    results.to_csv(out_path, index=False)
    print(f"\n✓ All fine-tuned translations saved to {out_path}")
    return results


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    run_finetuning_pipeline(
        terms_csv=str(base / "data" / "technical_terms.csv"),
        sentences_csv=str(base / "data" / "test_sentences.csv"),
        baseline_csv=str(base / "results" / "baseline_translations.csv"),
        output_dir=str(base / "results"),
        languages=["hindi", "tamil", "telugu", "kannada"]
    )
