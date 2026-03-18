"""
Baseline Translation Pipeline
Uses Helsinki-NLP opus-mt models from HuggingFace for multilingual translation.
These are general-purpose NMT models — we measure their accuracy on technical text.
"""

import pandas as pd
import time
from pathlib import Path
from typing import Optional


# Language configs: HuggingFace model name + language display name
LANGUAGE_CONFIGS = {
    'hindi': {
        'model': 'Helsinki-NLP/opus-mt-en-hi',
        'display': 'Hindi',
        'code': 'hi'
    },
    'tamil': {
        'model': 'Helsinki-NLP/opus-mt-en-dra',   # Dravidian family model
        'display': 'Tamil',
        'code': 'ta'
    },
    'telugu': {
        'model': 'Helsinki-NLP/opus-mt-en-dra',
        'display': 'Telugu',
        'code': 'te'
    },
    'kannada': {
        'model': 'Helsinki-NLP/opus-mt-en-dra',
        'display': 'Kannada',
        'code': 'kn'
    }
}


class BaselineTranslator:
    """
    Wraps HuggingFace translation pipelines for multiple target languages.
    Provides baseline (unadapted) translations for comparison.
    """

    def __init__(self, languages: list = None):
        self.languages = languages or list(LANGUAGE_CONFIGS.keys())
        self.pipelines = {}
        self._load_models()

    def _load_models(self):
        """Load HuggingFace translation pipelines."""
        try:
            from transformers import pipeline
            print("Loading translation models...")
            for lang in self.languages:
                config = LANGUAGE_CONFIGS[lang]
                print(f"  Loading {config['display']} model: {config['model']}")
                try:
                    self.pipelines[lang] = pipeline(
                        "translation",
                        model=config['model'],
                        src_lang="en",
                        tgt_lang=config['code']
                    )
                    print(f"  ✓ {config['display']} model loaded")
                except Exception as e:
                    print(f"  ✗ Failed to load {config['display']}: {e}")
                    self.pipelines[lang] = None
        except ImportError:
            print("transformers not installed. Run: pip install transformers sentencepiece")
            raise

    def translate(self, text: str, language: str) -> Optional[str]:
        """Translate a single sentence to the target language."""
        if language not in self.pipelines or self.pipelines[language] is None:
            return None
        try:
            result = self.pipelines[language](text, max_length=512)
            return result[0]['translation_text']
        except Exception as e:
            print(f"Translation error for {language}: {e}")
            return None

    def translate_batch(self, texts: list, language: str) -> list:
        """Translate a list of sentences."""
        if language not in self.pipelines or self.pipelines[language] is None:
            return [None] * len(texts)
        try:
            results = self.pipelines[language](texts, max_length=512, batch_size=8)
            return [r['translation_text'] for r in results]
        except Exception as e:
            print(f"Batch translation error for {language}: {e}")
            return [None] * len(texts)

    def translate_dataset(self, df: pd.DataFrame, source_col: str = 'english_sentence') -> pd.DataFrame:
        """
        Translate all sentences in a DataFrame for all configured languages.
        Returns DataFrame with new columns for each language's baseline translation.
        """
        result = df.copy()

        for lang in self.languages:
            config = LANGUAGE_CONFIGS[lang]
            col_name = f'baseline_{lang}'
            print(f"\nTranslating to {config['display']}...")

            start = time.time()
            texts = df[source_col].tolist()
            translations = self.translate_batch(texts, lang)
            elapsed = time.time() - start

            result[col_name] = translations
            success = sum(1 for t in translations if t is not None)
            print(f"  ✓ {success}/{len(texts)} sentences translated in {elapsed:.1f}s")

        return result


def run_baseline_translation(data_path: str, output_path: str):
    """
    Main function: load test sentences, run baseline translation, save results.
    """
    print("=" * 60)
    print("BASELINE TRANSLATION PIPELINE")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} test sentences from {data_path}")

    # Run translations
    translator = BaselineTranslator()
    results = translator.translate_dataset(df)

    # Save
    results.to_csv(output_path, index=False)
    print(f"\n✓ Baseline translations saved to {output_path}")

    # Preview
    print("\nSample output (first row):")
    row = results.iloc[0]
    print(f"  EN: {row['english_sentence']}")
    for lang in LANGUAGE_CONFIGS:
        col = f'baseline_{lang}'
        if col in results.columns:
            print(f"  {lang.upper()}: {row[col]}")

    return results


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "test_sentences.csv"
    output_path = Path(__file__).parent.parent / "results" / "baseline_translations.csv"
    output_path.parent.mkdir(exist_ok=True)

    run_baseline_translation(str(data_path), str(output_path))
