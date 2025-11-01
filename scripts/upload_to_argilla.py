import json
import argilla as rg

client = rg.Argilla(api_url="http://192.168.1.185:6900", api_key="argilla.apikey")

labels = ["STUDY_DESIGN", "SAMPLE_SIZE", "DEVICE", "AGE", "ADVERSE_EVENT", "OUTCOME"]

settings = rg.Settings(
    fields=[rg.TextField(name="text", title="Text", use_markdown=False)],
    questions=[
        rg.SpanQuestion(
            name="span_label",
            field="text",
            labels=labels,
            title="Select entity spans in the text",
            allow_overlapping=False
        )
    ],
    guidelines="Annotate entity spans in the text."
)

# Delete existing dataset and recreate
try:
    existing = client.datasets(name="ner_example_token_classification")
    existing.delete()
    print("🗑️ Deleted existing dataset")
except:
    pass

dataset = rg.Dataset(name="ner_example_token_classification", settings=settings)
dataset.create()
print("✅ Dataset created")

records = []
user_id = client.me.id

with open("assets/raw_data_autolabel.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        text = item["text"]
        
        # Create suggestions from predictions
        suggestions = []
        if item.get("prediction"):
            prediction_spans = [
                {
                    "start": int(pred["start"]), 
                    "end": int(pred["end"]), 
                    "label": str(pred["label"])
                } 
                for pred in item["prediction"]
            ]
            suggestions.append(
                rg.Suggestion(
                    question_name="span_label",
                    value=prediction_spans,
                    agent=item.get("prediction_agent", "model")
                )
            )
        
        # Create responses from annotations
        responses = []
        if item.get("annotation"):
            annotation_spans = [
                {
                    "start": int(ann["start"]), 
                    "end": int(ann["end"]), 
                    "label": str(ann["label"])
                } 
                for ann in item["annotation"]
            ]
            responses.append(
                rg.Response(
                    question_name="span_label",
                    value=annotation_spans,
                    user_id=user_id,
                    status="draft"
                )
            )
        
        record = rg.Record(
            fields={"text": text},
            suggestions=suggestions if suggestions else None,
            responses=responses if responses else None
        )
        records.append(record)

dataset.records.log(records)
print(f"✅ Uploaded {len(records)} records")