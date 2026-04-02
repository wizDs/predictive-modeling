"""
Lightweight NER-based skill highlighter for job postings.

Bootstraps training data from a skill taxonomy via phrase matching,
then trains a spaCy NER model that learns contextual patterns and
generalises beyond exact matches.

Categories:
  TECH      – programming languages, frameworks, methodologies
  TOOL      – specific platforms, products, services
  DOMAIN    – domain knowledge (e.g. risk management, NLP)
  SOFT      – soft / interpersonal skills
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

_HERE = Path(__file__).parent
MODEL_DIR = _HERE / "skill_ner_model"

TAXONOMY: dict[str, list[str]] = {
    "TECH": [
        # Languages
        "Python", "SQL", "SAS", "Java", "R", "Scala", "Go", "Rust",
        "JavaScript", "TypeScript", "C++", "C#", "Julia",
        # Methodologies / practices
        "DevOps", "MLOps", "CI/CD", "CI / CD", "DataOps",
        "machine learning", "deep learning", "reinforcement learning",
        "natural language processing", "NLP", "computer vision",
        "Generative AI", "GenAI", "LLM", "large language model",
        "AI/ML", "AI / ML", "AI", "ML",
        "data engineering", "data science", "data analytics",
        "containerisation", "containerization", "microservices",
        "event-driven", "event-based", "REST", "API",
        "GPU", "CUDA",
        # Danish equivalents
        "kunstig intelligens", "maskinlæring", "machine learning modeller",
    ],
    "TOOL": [
        "Azure", "AWS", "GCP", "Google Cloud",
        "GCP Vertex AI", "Vertex AI", "GCP BigQuery", "BigQuery",
        "Airflow", "Apache Airflow", "Apache NiFi",
        "Docker", "Kubernetes", "Ansible", "Terraform",
        "Grafana", "HAProxy", "OpenSearch", "Elasticsearch",
        "FastAPI", "vLLM", "Hugging Face", "HuggingFace",
        "Linux", "Git", "GitHub", "GitLab",
        "Spark", "Databricks", "dbt", "Snowflake", "Redshift",
        "PostgreSQL", "MongoDB", "Redis",
        "TensorFlow", "PyTorch", "scikit-learn", "spaCy", "pandas",
        "Raspberry Pi",
    ],
    "DOMAIN": [
        "risk management", "financial risk", "trading",
        "recommendation systems", "recommender systems",
        "sentiment analysis", "topic modeling", "topic modelling",
        "content generation", "personalisation", "personalization",
        "forecasting", "trend analysis", "ranking",
        "search", "information retrieval",
        "security", "cybersecurity", "intelligence",
        "efterretning", "spionage", "terror",
    ],
    "SOFT": [
        "collaboration", "teamwork", "stakeholder engagement",
        "communication", "problem solving", "problem-solving",
        "leadership", "mentoring", "coaching",
        "samarbejde", "videndeling", "sparring",
        "helhedsorienteret", "samarbejdsorienteret",
    ],
}

_LABEL_BY_PHRASE: dict[str, str] = {
    phrase.lower(): label
    for label, phrases in TAXONOMY.items()
    for phrase in phrases
}


def _find_spans(text: str, nlp: spacy.Language | None = None) -> list[tuple[int, int, str]]:
    """Return (start, end, label) spans found via taxonomy phrase matching.

    When *nlp* is given the spans are snapped to token boundaries so they
    align with spaCy's tokeniser (avoids W030 warnings during training).
    """
    spans: list[tuple[int, int, str]] = []
    text_lower = text.lower()

    # Build token boundary set for alignment
    tok_starts: set[int] = set()
    tok_ends: set[int] = set()
    if nlp is not None:
        doc = nlp.make_doc(text)
        for tok in doc:
            tok_starts.add(tok.idx)
            tok_ends.add(tok.idx + len(tok))

    for phrase, label in sorted(_LABEL_BY_PHRASE.items(), key=lambda x: -len(x[0])):
        for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text_lower):
            start, end = m.start(), m.end()
            # If we have token info, only keep spans on token boundaries
            if nlp is not None and (start not in tok_starts or end not in tok_ends):
                continue
            if any(s <= start < e or s < end <= e for s, e, _ in spans):
                continue
            spans.append((start, end, label))
    return spans


def annotate_text(text: str, nlp: spacy.Language | None = None) -> dict:
    """Return a spaCy-compatible training example dict."""
    spans = _find_spans(text, nlp=nlp)
    return {"text": text, "entities": spans}


def annotate_files(paths: Iterable[Path], nlp: spacy.Language | None = None) -> list[dict]:
    """Annotate multiple files and return training dicts."""
    if nlp is None:
        nlp = spacy.blank("xx")
    examples = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        for para in re.split(r"\n{2,}", text):
            para = para.strip()
            if len(para) < 20:
                continue
            ann = annotate_text(para, nlp=nlp)
            if ann["entities"]:
                examples.append(ann)
    return examples


def train(
    train_data: list[dict],
    output_dir: Path = MODEL_DIR,
    n_iter: int = 40,
    drop: float = 0.35,
) -> Path:
    """Train a spaCy NER model from annotated data and save it."""
    nlp = spacy.blank("xx")  # multilingual blank model
    ner = nlp.add_pipe("ner")
    for label in TAXONOMY:
        ner.add_label(label)

    # Convert to Example objects
    examples = []
    for item in train_data:
        doc = nlp.make_doc(item["text"])
        ents = {"entities": item["entities"]}
        examples.append(Example.from_dict(doc, ents))

    # Training loop
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(examples)
        losses: dict[str, float] = {}
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, drop=drop, sgd=optimizer, losses=losses)
        if (i + 1) % 10 == 0:
            print(f"  iter {i+1:>3}/{n_iter}  loss={losses.get('ner', 0):.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")
    return output_dir


def load_model(path: Path = MODEL_DIR) -> spacy.Language:
    return spacy.load(path)


def predict(text: str) -> list[dict]:
    """Return list of {start, end, label, text} dicts for detected skills."""
    nlp = load_model()
    doc = nlp(text)
    return [
        {"start": ent.start_char, "end": ent.end_char, "label": ent.label_, "text": ent.text}
        for ent in doc.ents
    ]


LABEL_COLOURS = {
    "TECH":   "#4dabf7",  # blue
    "TOOL":   "#69db7c",  # green
    "DOMAIN": "#ffd43b",  # yellow
    "SOFT":   "#da77f2",  # purple
}


def highlight_html(text: str, entities: list[dict]) -> str:
    """Return HTML with coloured <mark> spans around detected skills."""
    sorted_ents = sorted(entities, key=lambda e: e["start"])
    parts: list[str] = []
    prev = 0
    for ent in sorted_ents:
        s, e = ent["start"], ent["end"]
        colour = LABEL_COLOURS.get(ent["label"], "#adb5bd")
        parts.append(text[prev:s])
        parts.append(f'<mark style="background:{colour};padding:1px 4px;border-radius:3px;" title="{ent["label"]}">')
        parts.append(text[s:e])
        parts.append("</mark>")
        prev = e
    parts.append(text[prev:])
    return "".join(parts)
