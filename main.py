"""
Main Pipeline Runner
Orchestrates the full experiment:
  Step 1: Baseline translation
  Step 2: Glossary-based domain adaptation
  Step 3: Fine-tuning based domain adaptation
  Step 4: Evaluation and comparison
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

LANGUAGES = ['hindi', 'tamil', 'telugu', 'kannada']


def step1_baseline(args):
    print("\n" + "="*60)
    print("STEP 1: BASELINE TRANSLATION")
    print("="*60)
    from baseline_translator import run_baseline_translation
    RESULTS_DIR.mkdir(exist_ok=True)
    run_baseline_translation(
        data_path=str(DATA_DIR / "test_sentences.csv"),
        output_path=str(RESULTS_DIR / "baseline_translations.csv")
    )


def step2_glossary(args):
    print("\n" + "="*60)
    print("STEP 2: GLOSSARY-BASED ADAPTATION")
    print("="*60)
    from glossary_adapter import GlossaryAdapter

    baseline_path = RESULTS_DIR / "baseline_translations.csv"
    if not baseline_path.exists():
        print("Baseline translations not found. Run Step 1 first.")
        return

    df = pd.read_csv(baseline_path)
    result = df.copy()

    for language in LANGUAGES:
        col = f'baseline_{language}'
        if col not in df.columns:
            print(f"Skipping {language} — baseline column missing")
            continue

        adapter = GlossaryAdapter(str(DATA_DIR / "technical_terms.csv"), language)
        adapted_col = f'glossary_{language}'

        adapted_rows = []
        for _, row in df.iterrows():
            src = row.get('english_sentence', '')
            hyp = str(row[col]) if pd.notna(row[col]) else ''
            adapted, _ = adapter.adapt(src, hyp)
            adapted_rows.append(adapted)

        result[adapted_col] = adapted_rows
        print(f"✓ Glossary adaptation complete for {language}")

    out_path = RESULTS_DIR / "glossary_adapted_translations.csv"
    result.to_csv(out_path, index=False)
    print(f"\n✓ Glossary-adapted translations saved to {out_path}")


def step3_finetune(args):
    print("\n" + "="*60)
    print("STEP 3: FINE-TUNING ADAPTATION")
    print("="*60)
    from finetuning_adapter import run_finetuning_pipeline
    run_finetuning_pipeline(
        terms_csv=str(DATA_DIR / "technical_terms.csv"),
        sentences_csv=str(DATA_DIR / "test_sentences.csv"),
        baseline_csv=str(RESULTS_DIR / "baseline_translations.csv"),
        output_dir=str(RESULTS_DIR),
        languages=LANGUAGES
    )


def step4_evaluate(args):
    print("\n" + "="*60)
    print("STEP 4: EVALUATION")
    print("="*60)
    from evaluator import evaluate_all, save_evaluation_report, plot_comparison
    evaluation = evaluate_all(str(RESULTS_DIR), str(DATA_DIR))
    if evaluation:
        save_evaluation_report(evaluation, str(RESULTS_DIR / "evaluation_report"))
        plot_comparison(evaluation, str(RESULTS_DIR / "comparison_chart.png"))


def run_all(args):
    step1_baseline(args)
    step2_glossary(args)
    if not args.skip_finetune:
        step3_finetune(args)
    step4_evaluate(args)
    print("\n" + "="*60)
    print("✓ FULL PIPELINE COMPLETE")
    print(f"  Results in: {RESULTS_DIR}")
    print("="*60)


def demo(args):
    """
    Quick demo without downloading models — uses mock translations
    to demonstrate the glossary adaptation logic.
    """
    print("\n" + "="*60)
    print("DEMO: Glossary Adaptation (no model download required)")
    print("="*60)
    from glossary_adapter import GlossaryAdapter

    adapter = GlossaryAdapter(str(DATA_DIR / "technical_terms.csv"), 'hindi')

    test_cases = [
        ("The neural network uses backpropagation to update weights.",
         "The neural network uses backpropagation to update weights."),
        ("Gradient descent minimizes the loss function during training.",
         "Gradient descent minimizes the loss function during training."),
        ("Overfitting occurs when the model memorizes training data.",
         "Overfitting occurs when the model memorizes training data."),
        ("The transformer model applies an attention mechanism.",
         "The transformer model applies an attention mechanism."),
    ]

    print(f"\n{'Source (EN)':<55} {'Before Adaptation':<50} {'After Adaptation'}")
    print("-" * 160)

    for en, baseline in test_cases:
        adapted, replacements = adapter.adapt(en, baseline)
        print(f"{en[:53]:<55} {baseline[:48]:<50} {adapted}")
        if replacements:
            for en_term, target_term in replacements:
                print(f"  → '{en_term}' replaced with '{target_term}'")

    print("\n✓ Demo complete. Run 'python main.py --all' for the full pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Domain-Specific Translation Accuracy Improvement Pipeline"
    )
    subparsers = parser.add_subparsers()

    # Individual steps
    p1 = subparsers.add_parser('baseline', help='Run baseline translation')
    p1.set_defaults(func=step1_baseline)

    p2 = subparsers.add_parser('glossary', help='Run glossary adaptation')
    p2.set_defaults(func=step2_glossary)

    p3 = subparsers.add_parser('finetune', help='Run fine-tuning adaptation')
    p3.set_defaults(func=step3_finetune)

    p4 = subparsers.add_parser('evaluate', help='Run evaluation')
    p4.set_defaults(func=step4_evaluate)

    # Full pipeline
    p_all = subparsers.add_parser('all', help='Run full pipeline')
    p_all.add_argument('--skip-finetune', action='store_true',
                       help='Skip fine-tuning (faster, glossary only)')
    p_all.set_defaults(func=run_all)

    # Demo (no model download)
    p_demo = subparsers.add_parser('demo', help='Quick demo without downloading models')
    p_demo.set_defaults(func=demo)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python main.py demo          # No model download needed")
        print("  python main.py all           # Full pipeline")
        print("  python main.py all --skip-finetune  # Glossary only (faster)")
