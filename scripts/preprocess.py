import typer
import srsly
from pathlib import Path
from spacy.util import get_words_and_spaces
from spacy.tokens import Doc, DocBin
import spacy
import random


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    assets_dir: Path = typer.Argument(Path("assets"), exists=True, dir_okay=True),
    corpus_dir: Path = typer.Argument(Path("corpus"), exists=True, dir_okay=True),
    split_ratio: float = typer.Option(0.8, help="Proportion of data to use for training"),
    seed: int = typer.Option(42, help="Random seed for reproducible split"),
):
    """Split JSONL (with 'annotation' + 'status') into train/dev and convert to .spacy format."""

    random.seed(seed)
    nlp = spacy.blank("en")

    # 1️⃣ Read & filter validated examples
    examples = [eg for eg in srsly.read_jsonl(input_path) if eg.get("status") == "Validated"]
    random.shuffle(examples)

    split_point = int(len(examples) * split_ratio)
    train_examples = examples[:split_point]
    dev_examples = examples[split_point:]

    print(f"✅ Total validated examples: {len(examples)}")
    print(f"📘 Train: {len(train_examples)}, 📙 Dev: {len(dev_examples)}")

    assets_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # 2️⃣ Save split JSONLs to assets/
    train_jsonl = assets_dir / "train.jsonl"
    dev_jsonl = assets_dir / "dev.jsonl"
    srsly.write_jsonl(train_jsonl, train_examples)
    srsly.write_jsonl(dev_jsonl, dev_examples)
    print(f"💾 Saved JSONLs: {train_jsonl.name}, {dev_jsonl.name}")

    # 3️⃣ Convert JSONL -> .spacy in corpus/
    def convert_and_save(examples, output_file):
        doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
        for eg in examples:
            text = eg["text"]
            tokens = eg.get("tokens", [])
            words, spaces = get_words_and_spaces(tokens, text)
            doc = Doc(nlp.vocab, words=words, spaces=spaces)
            ents = []
            for ann in eg.get("annotation", []):
                span = doc.char_span(ann["start"], ann["end"], label=ann["label"])
                if span is not None:
                    ents.append(span)
            doc.ents = ents
            doc_bin.add(doc)
        doc_bin.to_disk(output_file)
        print(f"💾 Saved {len(doc_bin)} docs to {output_file}")

    convert_and_save(train_examples, corpus_dir / "train.spacy")
    convert_and_save(dev_examples, corpus_dir / "dev.spacy")


if __name__ == "__main__":
    typer.run(main)
