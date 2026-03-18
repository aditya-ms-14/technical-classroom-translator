"""
OPUS Dataset Downloader & Preprocessor
Downloads parallel corpora from OPUS (opus.nlpl.eu) via HuggingFace datasets library.

Sources used:
- OPUS-100 (multilingual, 100 languages, 1M sentence pairs per language)
- CCAligned (web-crawled parallel sentences)
- WikiMatrix (Wikipedia-based parallel sentences)

Languages: English → Hindi, Tamil, Telugu, Kannada
"""

import os
import csv
import json
import time
import random
import argparse
from pathlib import Path

# ── language configs ─────────────────────────────────────────────────────────
LANGUAGE_CONFIGS = {
    "hindi":   {"code": "hi", "hf_code": "hi", "opus_code": "hi"},
    "tamil":   {"code": "ta", "hf_code": "ta", "opus_code": "ta"},
    "telugu":  {"code": "te", "hf_code": "te", "opus_code": "te"},
    "kannada": {"code": "kn", "hf_code": "kn", "opus_code": "kn"},
}

# OPUS dataset names available on HuggingFace
OPUS_DATASETS = [
    {"name": "opus100",      "hf_path": "opus100",          "config_template": "en-{lang}"},
    {"name": "opus_books",   "hf_path": "Helsinki-NLP/opus_books", "config_template": "en-{lang}"},
    {"name": "CCAligned",    "hf_path": "Helsinki-NLP/ccaligned_multilingual", "config_template": "en_XX-{LANG}_XX"},
]


# ── helpers ──────────────────────────────────────────────────────────────────

def safe_load_dataset(hf_path: str, config: str, split: str = "train"):
    """Load a HuggingFace dataset, return None on failure."""
    try:
        from datasets import load_dataset
        print(f"    Trying {hf_path} [{config}] …", end=" ", flush=True)
        ds = load_dataset(hf_path, config, split=split, trust_remote_code=True)
        print(f"✓  ({len(ds):,} rows)")
        return ds
    except Exception as e:
        print(f"✗  ({e})")
        return None


def extract_pairs(dataset, lang_code: str, max_rows: int = 50_000) -> list[dict]:
    """
    Extract (english, target) pairs from a HuggingFace dataset row.
    Handles the different column schemas used across OPUS datasets.
    """
    pairs = []
    for row in dataset:
        try:
            # Schema 1: {'translation': {'en': '...', 'hi': '...'}}
            if "translation" in row:
                t = row["translation"]
                en = t.get("en", "").strip()
                tgt = t.get(lang_code, "").strip()
            # Schema 2: flat columns  en / hi  (or similar)
            elif "en" in row and lang_code in row:
                en  = str(row["en"]).strip()
                tgt = str(row[lang_code]).strip()
            # Schema 3: src / tgt
            elif "src" in row and "tgt" in row:
                en  = str(row["src"]).strip()
                tgt = str(row["tgt"]).strip()
            else:
                continue

            if en and tgt and en != tgt and len(en) > 10:
                pairs.append({"en": en, "target": tgt})
                if len(pairs) >= max_rows:
                    break
        except Exception:
            continue
    return pairs


def deduplicate(pairs: list[dict]) -> list[dict]:
    seen = set()
    out  = []
    for p in pairs:
        key = p["en"].lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def save_csv(pairs: list[dict], path: Path, lang: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["en", "target", "language"])
        writer.writeheader()
        for p in pairs:
            writer.writerow({"en": p["en"], "target": p["target"], "language": lang})
    print(f"  Saved {len(pairs):,} pairs → {path}")


def save_summary(summary: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved → {path}")


# ── per-language downloader ──────────────────────────────────────────────────

def download_for_language(language: str, output_dir: Path,
                           target_size: int = 10_000) -> dict:
    cfg      = LANGUAGE_CONFIGS[language]
    lang_code = cfg["code"]
    all_pairs: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  Downloading OPUS data for {language.upper()} ({lang_code})")
    print(f"  Target: {target_size:,} sentence pairs")
    print(f"{'='*60}")

    # ── 1. OPUS-100 (primary source, best quality) ────────────────────────
    config = f"en-{lang_code}"
    ds = safe_load_dataset("opus100", config)
    if ds is None:
        # Some language pairs are stored as {lang}-en instead of en-{lang}
        config = f"{lang_code}-en"
        ds = safe_load_dataset("opus100", config)
    if ds:
        pairs = extract_pairs(ds, lang_code, max_rows=target_size)
        print(f"  OPUS-100 extracted: {len(pairs):,} pairs")
        all_pairs.extend(pairs)

    # ── 2. Helsinki-NLP/opus_books (if still need more) ──────────────────
    if len(all_pairs) < target_size:
        for cfg_tmpl in [f"en-{lang_code}", f"{lang_code}-en"]:
            ds = safe_load_dataset("Helsinki-NLP/opus_books", cfg_tmpl)
            if ds:
                pairs = extract_pairs(ds, lang_code, max_rows=target_size - len(all_pairs))
                print(f"  OPUS Books extracted: {len(pairs):,} pairs")
                all_pairs.extend(pairs)
                break

    # ── 3. Helsinki-NLP/tatoeba_mt (sentence-level, diverse) ─────────────
    if len(all_pairs) < target_size:
        for cfg_tmpl in [f"eng-{lang_code}", f"{lang_code}-eng"]:
            ds = safe_load_dataset("Helsinki-NLP/tatoeba_mt", cfg_tmpl)
            if ds:
                pairs = extract_pairs(ds, lang_code, max_rows=target_size - len(all_pairs))
                print(f"  Tatoeba MT extracted: {len(pairs):,} pairs")
                all_pairs.extend(pairs)
                break

    # ── 4. WikiMatrix (Wikipedia parallel sentences) ──────────────────────
    if len(all_pairs) < target_size:
        for cfg_tmpl in [f"en-{lang_code}", f"{lang_code}-en"]:
            ds = safe_load_dataset("Helsinki-NLP/wikimatrix", cfg_tmpl)
            if ds:
                pairs = extract_pairs(ds, lang_code, max_rows=target_size - len(all_pairs))
                print(f"  WikiMatrix extracted: {len(pairs):,} pairs")
                all_pairs.extend(pairs)
                break

    # ── Deduplicate & trim ────────────────────────────────────────────────
    all_pairs = deduplicate(all_pairs)
    random.shuffle(all_pairs)
    all_pairs = all_pairs[:target_size]

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = output_dir / f"opus_{language}.csv"
    if all_pairs:
        save_csv(all_pairs, out_path, language)
    else:
        print(f"  ⚠  No pairs collected for {language}. "
              f"The language may not be available in OPUS via HuggingFace.")

    return {
        "language": language,
        "lang_code": lang_code,
        "pairs_collected": len(all_pairs),
        "output_file": str(out_path) if all_pairs else None,
    }


# ── merge with existing glossary sentences ───────────────────────────────────

def merge_with_glossary(opus_path: Path, glossary_terms_path: Path,
                         language: str, merged_path: Path):
    """
    Combine OPUS data with the hand-crafted technical glossary pairs
    so the model always sees the correct domain translations.
    """
    import pandas as pd

    rows = []

    # Load OPUS pairs
    if opus_path.exists():
        df_opus = pd.read_csv(opus_path)
        rows.append(df_opus[["en", "target"]])
        print(f"  OPUS rows:    {len(df_opus):,}")

    # Build glossary pairs
    if glossary_terms_path.exists():
        df_terms = pd.read_csv(glossary_terms_path)
        if language in df_terms.columns:
            gloss_pairs = df_terms[["term", language]].rename(
                columns={"term": "en", language: "target"}
            )
            rows.append(gloss_pairs)
            print(f"  Glossary rows: {len(gloss_pairs)}")

    if not rows:
        print("  Nothing to merge.")
        return

    merged = pd.concat(rows, ignore_index=True).drop_duplicates(subset="en")
    merged["language"] = language
    merged.to_csv(merged_path, index=False)
    print(f"  Merged total: {len(merged):,} → {merged_path}")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download OPUS parallel corpora for Hindi/Tamil/Telugu/Kannada"
    )
    parser.add_argument(
        "--languages", nargs="+",
        default=list(LANGUAGE_CONFIGS.keys()),
        help="Languages to download (default: all four)"
    )
    parser.add_argument(
        "--target-size", type=int, default=10_000,
        help="Sentence pairs per language (default: 10000)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <project>/data/opus)"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge OPUS data with existing technical glossary"
    )
    args = parser.parse_args()

    base_dir    = Path(__file__).parent.parent
    output_dir  = Path(args.output_dir) if args.output_dir else base_dir / "data" / "opus"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OPUS DATASET DOWNLOADER")
    print(f"Languages : {', '.join(args.languages)}")
    print(f"Target    : {args.target_size:,} pairs per language")
    print(f"Output    : {output_dir}")
    print("=" * 60)

    summary   = {}
    t0        = time.time()

    for language in args.languages:
        result = download_for_language(language, output_dir,
                                        target_size=args.target_size)
        summary[language] = result

        if args.merge and result["output_file"]:
            print(f"\n  Merging with glossary for {language}…")
            merge_with_glossary(
                opus_path          = Path(result["output_file"]),
                glossary_terms_path= base_dir / "data" / "technical_terms.csv",
                language           = language,
                merged_path        = output_dir / f"merged_{language}.csv",
            )

    elapsed = time.time() - t0
    save_summary(summary, output_dir / "download_summary.json")

    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    for lang, info in summary.items():
        status = f"{info['pairs_collected']:,} pairs" if info["pairs_collected"] else "FAILED"
        print(f"  {lang:<10} {status}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Files in  : {output_dir}")
    print("\nNext step:")
    print("  python main.py all   # runs pipeline with the new OPUS data")


if __name__ == "__main__":
    main()
