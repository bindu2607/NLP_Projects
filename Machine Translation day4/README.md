# Translation Evaluation Pipeline

## Overview

This project evaluates the performance of an English-to-French translation pipeline by translating ASR-generated text and comparing the results against ground truth translations using BLEU scores.

Key functionalities include:
- Loading ASR outputs and reference translations from JSON.
- Translating inputs with the MarianMT model via Hugging Face.
- Displaying per-sample translation alongside the ground truth with match status.
- Computing corpus-level BLEU score using SacreBLEU.
- Saving detailed evaluation results to a JSON file.

---

## Setup Instructions

1. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```
2. Install required packages:
```
pip install -r requirements.txt
```
3. Ensure data/test_samples.json exists with entries like:
json
[
  {
    "id": 1,
    "asr_output": "Hello how are you today?",
    "ground_truth_translation": "Bonjour, comment allez-vous aujourd'hui ?"
  },
  {
    "id": 2,
    "asr_output": "The weather is nice outside.",
    "ground_truth_translation": "Il fait beau dehors."
  }
  // more samples...
]
4. Run the evaluation script
```
python src/evaluate_translations.py
```
The script will:

- Load test samples.

- Translate ASR outputs with MarianMT English-to-French model.

- Print each translation and whether it matches the ground truth.

- Compute and display the overall BLEU score.

- Save full detailed results to outputs/evaluation_results.json.

## Sample Output
```
--- Translation Evaluation Results ---
Source (ASR): "Hello how are you today?"
Translated  : "Bonjour, comment allez-vous aujourd'hui ?"
Expected    : "Bonjour, comment allez-vous aujourd'hui ?"
Match       : Yes
```
...

--- Summary ---
Overall BLEU Score: 73.03
Detailed results saved to: outputs/evaluation_results.json
-----------------