#!/usr/bin/env python3
"""Train the skill NER model on existing job postings."""

import sys
from pathlib import Path

from wiz.job_app_backend import annotate_files, train, MODEL_DIR, load_model


def main(data_dir: Path) -> None:
    postings = sorted(data_dir.glob("*/job_posting.tex"))
    if not postings:
        print(f"No job postings found in {data_dir}/*/job_posting.tex")
        return

    print(f"Found {len(postings)} job posting(s):")
    for p in postings:
        print(f"  {p.relative_to(data_dir)}")

    print("\nAnnotating...")
    train_data = annotate_files(postings)
    print(f"  {len(train_data)} annotated paragraphs")

    for item in train_data[:3]:
        ents = item["entities"]
        print(f"  [{len(ents)} entities] {item['text'][:80]}...")

    print("\nTraining...")
    train(train_data, output_dir=MODEL_DIR, n_iter=40)

    print("\nSample predictions:")
    nlp = load_model()
    for item in train_data[:3]:
        doc = nlp(item["text"])
        for ent in doc.ents:
            print(f"  {ent.label_:8s}  {ent.text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data_dir>")
        print("  e.g. python -m wiz.job_app_backend.train_model ./apps/tools-app/job-app/data")
        sys.exit(1)
    main(Path(sys.argv[1]))
