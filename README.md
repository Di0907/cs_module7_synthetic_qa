# Module 7 — Synthetic QA Dataset & Fine-Tuning

## Description (What this homework is about)

This assignment focuses on generating a synthetic question–answer (QA) dataset using a large language model and exploring how such data can be used for downstream fine-tuning and evaluation.

The workflow includes:
1. Generating synthetic QA pairs from academic-style paper abstracts.
2. Structuring the data in JSONL format suitable for training or fine-tuning.
3. Providing a fine-tuning script that demonstrates how the dataset can be consumed.
4. Evaluating QA accuracy by comparing model responses before and after fine-tuning (or with synthetic supervision).

The goal of this homework is to understand data-centric approaches to improving QA systems and to practice building a clean, reproducible ML pipeline.

---

## Repository Structure

```text
module7/
├── data/
│   └── synthetic_qa.jsonl        # Synthetic QA dataset (JSONL)
├── scripts/
│   ├── generate_dataset.py       # Script to generate synthetic QA pairs
│   └── finetune.py               # Fine-tuning script
├── report/
│   └── Evaluation Report.pdf     # QA accuracy evaluation report (≤ 1 page)
└── README.md
```
## Deliverables

1. Synthetic dataset in JSONL format
Location: data/synthetic_qa.jsonl
Each line contains an instruction, input context, expected output, and paper ID.

2. Fine-tuning script
Location: scripts/finetune.py
Demonstrates how the synthetic QA dataset can be loaded and used for fine-tuning or supervised training.

3. Evaluation report comparing QA accuracy
Location: report/Evaluation Report.pdf
Provides a concise comparison of QA performance and qualitative analysis (≤ 1 page).

## Primary Reviewer
Primary Reviewer: Scott Lai

## Questions
1. How does synthetic QA data affect model performance compared to zero-shot prompting?
2. What types of questions benefit the most from synthetic supervision?
3. What limitations arise when generating QA pairs from short or incomplete abstracts?

## How to Run
Generate dataset:
python scripts/generate_dataset.py

Run fine-tuning:
python scripts/finetune.py

## Notes
1. All code is written in Python.
2. The dataset follows JSONL format for compatibility with common fine-tuning pipelines.
3. This repository is intended for educational purposes.
