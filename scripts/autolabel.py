import spacy
import json
from pathlib import Path

# === paths ===
MODEL_PATH = Path("training/model-best")
UNLABELED_PATH = Path("assets/unlabeled_sentences.txt")
OUTPUT_PATH = Path("assets/raw_data_autolabel.jsonl")

# === load model ===
print(f"Loading model from {MODEL_PATH} ...")
nlp = spacy.load(MODEL_PATH)

# === read unlabeled sentences ===
with open(UNLABELED_PATH, "r", encoding="utf8") as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"Read {len(texts)} unlabeled sentences")

# === run model and create predictions ===
with open(OUTPUT_PATH, "w", encoding="utf8") as fout:
    for text in texts:
        doc = nlp(text)
        
        # Build prediction list with start, end, label, and score
        predictions = [
            {
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "score": 1.0  # spaCy doesn't provide scores by default, using 1.0
            }
            for ent in doc.ents
        ]
        
        # Build annotation list (same format as prediction for weak labeling)
        annotations = [
            {
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            }
            for ent in doc.ents
        ]
        
        # Build tokens list with capitalness detection
        tokens = []
        for token in doc:
            # Determine capitalness
            if token.text.isupper():
                capitalness = "UPPER"
            elif token.text[0].isupper() and len(token.text) > 1:
                if token.text[1:].islower():
                    capitalness = "FIRST"
                else:
                    capitalness = "MIDDLE"
            elif token.text.islower():
                capitalness = "LOWER"
            else:
                capitalness = None
            
            tokens.append(token.text)
        
        # Build the complete record matching your format
        record = {
            "text": text,
            "tokens": tokens,
            "prediction": predictions,
            "prediction_agent": "Model/model-best",
            "annotation": annotations,
            "annotation_agent": "model-best",
            "vectors": None,
            "id": None,  # You can generate UUIDs if needed
            "metadata": None,
            "status": "WeaklyAnnotated",
            "event_timestamp": None,
            "metrics": {
                "annotated": {
                    "mentions": [
                        {
                            "capitalness": doc[ent.start].text[0].isupper() and "FIRST" or "LOWER",
                            "label": ent.label_,
                            "score": 1.0,
                            "value": ent.text
                        }
                        for ent in doc.ents
                    ]
                },
                "predicted": {
                    "mentions": [
                        {
                            "capitalness": doc[ent.start].text[0].isupper() and "FIRST" or "LOWER",
                            "label": ent.label_,
                            "score": 1.0,
                            "value": ent.text
                        }
                        for ent in doc.ents
                    ]
                },
                "text_length": len(text),
                "tokens": [
                    {
                        "capitalness": "UPPER" if t.text.isupper() 
                                     else "FIRST" if (t.text[0].isupper() and len(t.text) > 1 and t.text[1:].islower())
                                     else "MIDDLE" if (t.text[0].isupper() and len(t.text) > 1)
                                     else "LOWER" if t.text.islower()
                                     else None,
                        "value": t.text
                    }
                    for t in doc
                ]
            }
        }
        
        fout.write(json.dumps(record) + "\n")

print(f"✅ Generated {len(texts)} weakly annotated samples in {OUTPUT_PATH}")