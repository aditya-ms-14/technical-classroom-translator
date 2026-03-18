"""
Glossary-Based Domain Adaptation for Technical Translation
Approach 1: Post-edit translations using a curated technical glossary
"""

import pandas as pd
import re
import json
from pathlib import Path


class GlossaryAdapter:
    """
    Applies a technical glossary to correct mistranslations
    produced by general-purpose NMT models.
    """

    def __init__(self, glossary_path: str, language: str):
        """
        Args:
            glossary_path: Path to technical_terms.csv
            language: Target language code ('hindi', 'tamil', 'telugu', 'kannada')
        """
        self.language = language
        self.glossary = self._load_glossary(glossary_path, language)

    def _load_glossary(self, path: str, language: str) -> dict:
        df = pd.read_csv(path)
        glossary = {}
        for _, row in df.iterrows():
            term_en = row['term'].lower()
            term_target = row[language]
            glossary[term_en] = term_target
        # Sort by length descending so longer phrases match first
        glossary = dict(sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True))
        return glossary

    def adapt(self, english_text: str, translated_text: str) -> str:
        """
        Post-edit a translation by replacing mistranslated technical terms
        with correct glossary entries.

        Strategy:
        - Find each English technical term in the source
        - Check if the baseline translation preserved the English term (common for NMT)
        - Replace with correct target-language term from glossary
        """
        adapted = translated_text
        replacements_made = []

        for en_term, target_term in self.glossary.items():
            # Pattern 1: NMT kept the English term as-is (transliterated or unchanged)
            # We replace it with the proper term
            pattern_en = re.compile(re.escape(en_term), re.IGNORECASE)
            if pattern_en.search(adapted):
                adapted = pattern_en.sub(target_term, adapted)
                replacements_made.append((en_term, target_term))

        return adapted, replacements_made

    def batch_adapt(self, df: pd.DataFrame, translation_col: str, source_col: str = 'english_sentence') -> pd.DataFrame:
        """Apply glossary adaptation to a DataFrame of translations."""
        adapted_translations = []
        all_replacements = []

        for _, row in df.iterrows():
            adapted, replacements = self.adapt(row[source_col], row[translation_col])
            adapted_translations.append(adapted)
            all_replacements.append(replacements)

        result = df.copy()
        result[f'{translation_col}_glossary_adapted'] = adapted_translations
        result[f'{translation_col}_replacements'] = all_replacements
        return result

    def get_glossary_coverage(self, text: str) -> list:
        """Return list of technical terms found in a given English text."""
        found = []
        text_lower = text.lower()
        for term in self.glossary:
            if term in text_lower:
                found.append(term)
        return found


def load_glossary_as_json(glossary_path: str, output_path: str):
    """Export glossary to JSON for easy reference."""
    df = pd.read_csv(glossary_path)
    languages = ['hindi', 'tamil', 'telugu', 'kannada']
    glossary_dict = {}

    for _, row in df.iterrows():
        term = row['term']
        glossary_dict[term] = {
            'definition': row['definition'],
            'domain': row['domain'],
            'translations': {lang: row[lang] for lang in languages}
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(glossary_dict, f, ensure_ascii=False, indent=2)

    print(f"Glossary exported to {output_path} ({len(glossary_dict)} terms)")
    return glossary_dict


if __name__ == "__main__":
    glossary_path = Path(__file__).parent.parent / "data" / "technical_terms.csv"
    output_path = Path(__file__).parent.parent / "data" / "glossary.json"

    # Export glossary to JSON
    glossary = load_glossary_as_json(str(glossary_path), str(output_path))

    # Demo
    adapter = GlossaryAdapter(str(glossary_path), language='hindi')
    sample_en = "The neural network uses backpropagation to minimize the loss function."
    sample_translated = "The neural network uses backpropagation to minimize the loss function."  # NMT kept English terms

    adapted, replacements = adapter.adapt(sample_en, sample_translated)
    print(f"\nOriginal (EN): {sample_en}")
    print(f"Baseline:      {sample_translated}")
    print(f"Adapted:       {adapted}")
    print(f"Replacements:  {replacements}")

    # Show glossary coverage
    coverage = adapter.get_glossary_coverage(sample_en)
    print(f"Terms found in sentence: {coverage}")
