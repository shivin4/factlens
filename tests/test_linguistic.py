"""
Tests for Stage 1: Linguistic Analysis
"""
import pytest
from pipeline.linguistic import analyze, extract_claims, extract_concepts


def test_analyze_returns_all_fields():
    text = "BERT uses bidirectional attention for language modelling."
    result = analyze(text)
    assert isinstance(result.sentences, list)
    assert isinstance(result.entities, list)
    assert isinstance(result.concepts, list)
    assert isinstance(result.claims, list)


def test_sentences_split_correctly():
    text = "BERT is a model. Word2Vec uses skip-gram."
    result = analyze(text)
    assert len(result.sentences) == 2


def test_empty_input():
    result = analyze("")
    assert result.sentences == []
    assert result.claims == []
    assert result.concepts == []


def test_tech_term_ner():
    text = "BERT and Word2Vec are commonly used NLP models."
    result = analyze(text)
    entity_texts = [e.text for e in result.entities]
    # At least one of the known tech terms should be found
    found = any(t in entity_texts for t in ["BERT", "Word2Vec", "NLP"])
    assert found, f"No known tech terms found in: {entity_texts}"


def test_concepts_are_unique():
    text = "The transformer model uses the attention mechanism. The attention mechanism is important."
    result = analyze(text)
    assert len(result.concepts) == len(set(result.concepts))


def test_svo_extraction_simple():
    text = "BERT uses bidirectional attention."
    result = analyze(text)
    # Should detect at least one SVO claim
    assert len(result.claims) >= 1


def test_to_dict_serializable():
    import json
    text = "The Transformer model relies on self-attention."
    result = analyze(text)
    d = result.to_dict()
    # Should be JSON-serializable
    json_str = json.dumps(d)
    assert len(json_str) > 0
