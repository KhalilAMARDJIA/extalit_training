#!/usr/bin/env python3
"""
Export Argilla NER dataset and convert to legacy JSONL format.

This script:
1. Extracts the "ner_example_token_classification" dataset from Argilla
2. Saves raw records to assets/records.json
3. Filters only completed annotations
4. Converts to legacy JSONL format in assets/from_argilla.jsonl
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argilla as rg


def ensure_directories():
    """Create necessary directories if they don't exist."""
    Path("assets").mkdir(exist_ok=True)


def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization function.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens
    """
    # Simple whitespace tokenization (you can enhance this with spaCy if needed)
    tokens = text.split()
    return tokens


def calculate_capitalness(token: str) -> str:
    """
    Determine the capitalness of a token.
    
    Args:
        token: Input token
        
    Returns:
        Capitalness category: "FIRST", "UPPER", "LOWER", or None
    """
    if not token or not token.isalpha():
        return None
    if token[0].isupper() and token[1:].islower():
        return "FIRST"
    elif token.isupper():
        return "UPPER"
    elif token.islower():
        return "LOWER"
    return None


def char_to_token_spans(text: str, tokens: List[str], char_spans: List[tuple]) -> List[tuple]:
    """
    Convert character-based spans to token-based spans.
    
    Args:
        text: Original text
        tokens: List of tokens
        char_spans: List of (label, start_char, end_char) tuples
        
    Returns:
        List of (label, start_token, end_token) tuples
    """
    token_spans = []
    
    # Build character to token mapping
    char_to_token = []
    current_pos = 0
    
    for token_idx, token in enumerate(tokens):
        # Find token in text starting from current position
        token_start = text.find(token, current_pos)
        if token_start == -1:
            continue
        token_end = token_start + len(token)
        
        for char_idx in range(token_start, token_end):
            if char_idx >= len(char_to_token):
                char_to_token.extend([None] * (char_idx - len(char_to_token) + 1))
            char_to_token[char_idx] = token_idx
        
        current_pos = token_end
    
    # Convert character spans to token spans
    for label, start_char, end_char in char_spans:
        if start_char < len(char_to_token) and end_char <= len(char_to_token):
            start_token = char_to_token[start_char] if start_char < len(char_to_token) else None
            end_token = char_to_token[end_char - 1] if end_char > 0 and end_char - 1 < len(char_to_token) else None
            
            if start_token is not None and end_token is not None:
                token_spans.append((label, start_token, end_token + 1))
    
    return token_spans


def convert_argilla_to_legacy(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Argilla record format to legacy JSONL format.
    
    Args:
        record: Argilla record dictionary
        
    Returns:
        Legacy format dictionary
    """
    # Extract text
    text = record["fields"]["text"]
    
    # Tokenize
    tokens = tokenize_text(text)
    
    # Extract suggestions (predictions)
    prediction = []
    prediction_agent = "Model/model-best"
    
    if "suggestions" in record and "span_label" in record["suggestions"]:
        suggestion = record["suggestions"]["span_label"]
        if suggestion.get("value"):
            for span in suggestion["value"]:
                prediction.append({
                    "label": span["label"],
                    "start": span["start"],
                    "end": span["end"]
                })
        if suggestion.get("agent"):
            prediction_agent = suggestion["agent"]
    
    # Extract annotations (responses)
    annotation = []
    annotation_agent = "annotator"
    
    if "responses" in record and "span_label" in record["responses"]:
        responses = record["responses"]["span_label"]
        if responses and len(responses) > 0:
            response = responses[0]  # Take first response
            if "value" in response and response["value"]:
                for span in response["value"]:
                    annotation.append({
                        "label": span["label"],
                        "start": span["start"],
                        "end": span["end"]
                    })
            if "user_id" in response:
                annotation_agent = response["user_id"]
    
    # Convert character spans to mention format
    def spans_to_mentions(spans, text, tokens):
        mentions = []
        for span_dict in spans:
            label = span_dict["label"]
            start_char = span_dict["start"]
            end_char = span_dict["end"]
            value = text[start_char:end_char]
            
            # Determine capitalness of first token in span
            first_word = value.split()[0] if value.split() else value
            capitalness = calculate_capitalness(first_word)
            
            mentions.append({
                "capitalness": capitalness,
                "label": label,
                "score": span_dict.get("score", 1.0),
                "value": value
            })
        return mentions
    
    # Build metrics
    metrics = {
        "annotated": {
            "mentions": spans_to_mentions(annotation, text, tokens)
        },
        "predicted": {
            "mentions": spans_to_mentions(prediction, text, tokens)
        },
        "text_length": len(text),
        "tokens": [
            {
                "capitalness": calculate_capitalness(token),
                "value": token
            }
            for token in tokens
        ]
    }
    
    # Build legacy format
    legacy_record = {
        "text": text,
        "tokens": tokens,
        "prediction": [(span["label"], span["start"], span["end"]) for span in prediction],
        "prediction_agent": prediction_agent,
        "annotation": [(span["label"], span["start"], span["end"]) for span in annotation],
        "annotation_agent": annotation_agent,
        "vectors": record.get("vectors"),
        "id": record["id"],
        "metadata": record.get("metadata"),
        "status": record.get("status", "pending").capitalize(),
        "event_timestamp": None,  # Not available in new format
        "metrics": metrics
    }
    
    return legacy_record


def main():
    """Main execution function."""
    # Ensure directories exist
    ensure_directories()
    
    # Connect to Argilla
    # You may need to set these environment variables or pass them explicitly:
    # ARGILLA_API_URL and ARGILLA_API_KEY
    try:
        client = rg.Argilla(
            api_url=os.getenv("ARGILLA_API_URL", "http://192.168.1.185:6900"),
            api_key=os.getenv("ARGILLA_API_KEY", "argilla.apikey")
        )
        print(f"✓ Connected to Argilla at {client.api_url}")
    except Exception as e:
        print(f"✗ Failed to connect to Argilla: {e}")
        print("Please set ARGILLA_API_URL and ARGILLA_API_KEY environment variables")
        return
    
    # Get dataset
    dataset_name = "ner_example_token_classification"
    try:
        dataset = client.datasets(name=dataset_name)
        print(f"✓ Found dataset: {dataset_name}")
    except Exception as e:
        print(f"✗ Failed to load dataset '{dataset_name}': {e}")
        return
    
    # Fetch all records
    print("Fetching records from Argilla...")
    all_records = []
    
    try:
        record_count = 0
        for record in dataset.records:
            record_count += 1
            
            # Debug: Print first record structure
            if record_count == 1:
                print(f"\n[DEBUG] First record structure:")
                print(f"  - Type: {type(record)}")
                print(f"  - ID: {record.id}")
                print(f"  - Fields type: {type(record.fields)}")
                print(f"  - Fields: {record.fields}")
                if hasattr(record, 'suggestions'):
                    print(f"  - Has suggestions: {len(list(record.suggestions)) if record.suggestions else 0}")
                if hasattr(record, 'responses'):
                    print(f"  - Has responses: {len(list(record.responses)) if record.responses else 0}")
                print()
            
            # Convert record to dictionary format
            record_dict = {
                "id": str(record.id),
                "fields": dict(record.fields) if hasattr(record.fields, '__iter__') else record.fields,
                "metadata": dict(record.metadata) if record.metadata and hasattr(record.metadata, '__iter__') else {},
                "suggestions": {},
                "responses": {},
                "vectors": {},
                "status": record.status,
                "_server_id": str(record.id)
            }
            
            # Add suggestions
            if hasattr(record, 'suggestions') and record.suggestions:
                for suggestion in record.suggestions:
                    question_name = suggestion.question_name if hasattr(suggestion, 'question_name') else 'span_label'
                    
                    if suggestion.value:
                        # Handle different value formats
                        span_values = []
                        if hasattr(suggestion.value, '__iter__') and not isinstance(suggestion.value, str):
                            for span in suggestion.value:
                                span_values.append({
                                    "label": span.label if hasattr(span, 'label') else span.get('label'),
                                    "start": span.start if hasattr(span, 'start') else span.get('start'),
                                    "end": span.end if hasattr(span, 'end') else span.get('end'),
                                    "score": getattr(span, 'score', None) if hasattr(span, 'score') else span.get('score')
                                })
                        
                        record_dict["suggestions"][question_name] = {
                            "value": span_values,
                            "score": suggestion.score if hasattr(suggestion, 'score') else None,
                            "agent": suggestion.agent if hasattr(suggestion, 'agent') else "Model/model-best"
                        }
            
            # Add responses
            if hasattr(record, 'responses') and record.responses:
                for response in record.responses:
                    question_name = response.question_name if hasattr(response, 'question_name') else 'span_label'
                    
                    if question_name not in record_dict["responses"]:
                        record_dict["responses"][question_name] = []
                    
                    if response.value:
                        # Handle different value formats
                        span_values = []
                        if hasattr(response.value, '__iter__') and not isinstance(response.value, str):
                            for span in response.value:
                                span_values.append({
                                    "label": span.label if hasattr(span, 'label') else span.get('label'),
                                    "start": span.start if hasattr(span, 'start') else span.get('start'),
                                    "end": span.end if hasattr(span, 'end') else span.get('end')
                                })
                        
                        record_dict["responses"][question_name].append({
                            "value": span_values,
                            "user_id": str(response.user_id) if hasattr(response, 'user_id') else "annotator"
                        })
            
            all_records.append(record_dict)
        
        print(f"✓ Fetched {len(all_records)} records")
    except Exception as e:
        print(f"✗ Error fetching records: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save all records to JSON
    records_path = "assets/records.json"
    try:
        with open(records_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved all records to {records_path}")
    except Exception as e:
        print(f"✗ Failed to save records: {e}")
        return
    
    # Filter completed records only
    completed_records = [r for r in all_records if r.get("status") == "completed"]
    print(f"✓ Found {len(completed_records)} completed records")
    
    if not completed_records:
        print("⚠ No completed records found. Nothing to convert.")
        return
    
    # Convert to legacy format
    print("Converting to legacy JSONL format...")
    legacy_records = []
    
    for record in completed_records:
        try:
            legacy_record = convert_argilla_to_legacy(record)
            legacy_records.append(legacy_record)
        except Exception as e:
            print(f"⚠ Error converting record {record.get('id')}: {e}")
            continue
    
    # Save to JSONL
    output_path = "assets/from_argilla.jsonl"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for record in legacy_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✓ Saved {len(legacy_records)} records to {output_path}")
    except Exception as e:
        print(f"✗ Failed to save JSONL: {e}")
        return
    
    print("\n✅ Export complete!")
    print(f"   - All records: {records_path}")
    print(f"   - Completed (JSONL): {output_path}")


if __name__ == "__main__":
    main()