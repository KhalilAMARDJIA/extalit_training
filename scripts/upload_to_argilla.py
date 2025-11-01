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

dataset = rg.Dataset(name="ner_example_token_classification", settings=settings)
dataset.create()
print("✅ Dataset created")

records = []
user_id = client.me.id  # Use your Argilla account's user ID

with open("assets/raw_data_autolabel.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        text = item["text"]
        spans = [{"start": ann["start"], "end": ann["end"], "label": ann["label"]} 
                 for ann in item.get("annotation", [])]

        responses = [
            rg.Response(
                question_name="span_label",
                value=spans,
                user_id=user_id
            )
        ] if spans else []

        record = rg.Record(fields={"text": text}, responses=responses)
        records.append(record)

dataset.records.log(records)
print(f"✅ Uploaded {len(records)} records")
