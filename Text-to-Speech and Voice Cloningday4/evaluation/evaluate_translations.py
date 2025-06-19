import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from datetime import datetime
from sacrebleu import corpus_bleu
from src.translator import translate_text
import string

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def evaluate_translations():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'test_samples.json')
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find test_samples.json at {data_path}")
        return

    model_translations = []
    ground_truths = []
    results_data = []

    print("--- Translation Evaluation Results ---")
    for sample in test_samples:
        asr_input = sample['asr_output']
        ground_truth = sample['ground_truth_translation']
        model_output = translate_text(asr_input)
        model_translations.append(model_output)
        ground_truths.append(ground_truth)
        results_data.append({
            'id': sample['id'],
            'input_asr': asr_input,
            'ground_truth_translation': ground_truth,
            'model_translation': model_output
        })
        match = 'Yes' if normalize_text(model_output) == normalize_text(ground_truth) else 'No'
        print(f"Source (ASR): \"{asr_input}\"")
        print(f"Translated  : \"{model_output}\"")
        print(f"Expected    : \"{ground_truth}\"")
        print(f"Match       : {match}\n")
    bleu = corpus_bleu(model_translations, [ground_truths])
    final_output = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'bleu_score': f"{bleu.score:.2f}",
        'results': results_data
    }
    output_filename = os.path.join(output_dir, 'evaluation_results.json')
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print("--- Summary ---")
    print(f"Overall BLEU Score: {bleu.score:.2f}")
    print(f"Detailed results saved to: {output_filename}")
    print("-----------------")

if __name__ == "__main__":
    evaluate_translations()
