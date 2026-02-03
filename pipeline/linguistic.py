"""
FactLens — Stage 1: Linguistic Analysis
=========================================
Input : Raw text string (user's explanation)
Output: {
    "sentences" : [str, ...],
    "entities"  : [{text, label, start, end}, ...],
    "concepts"  : [str, ...],          # unique noun phrases (lemmatized)
    "claims"    : [{subject, predicate, object, raw_sentence}, ...]
}

Uses spaCy en_core_web_trf for high-accuracy parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

import spacy
from spacy.tokens import Doc, Token

# ── Load model (cached on first call) ─────────────────────────────────────────
_NLP: Optional[spacy.Language] = None


def _get_nlp() -> spacy.Language:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_trf")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_trf' not found.\n"
                "Run: python -m spacy download en_core_web_trf"
            )
    return _NLP


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass
class Claim:
    subject: str
    predicate: str
    object: str
    raw_sentence: str

    def as_sentence(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


@dataclass
class LinguisticOutput:
    sentences: list[str]
    entities: list[Entity]
    concepts: list[str]
    claims: list[Claim]

    def to_dict(self) -> dict:
        return {
            "sentences": self.sentences,
            "entities": [asdict(e) for e in self.entities],
            "concepts": self.concepts,
            "claims": [asdict(c) for c in self.claims],
        }


# ── NER labels considered domain-relevant ─────────────────────────────────────
DOMAIN_ENTITY_LABELS = {
    "ORG",      # organizations / research groups
    "PRODUCT",  # model names (BERT, GPT, etc.)
    "WORK_OF_ART",  # paper titles
    "PERSON",   # researchers
    "EVENT",
    "FAC",
    "NORP",
    "GPE",
    "LOC",
}

# Also captures technical terms not caught by NER
TECHNICAL_NER_PATTERN = re.compile(
    r'\b(BERT|GPT|ELMo|GloVe|word2vec|FastText|Transformer|RNN|LSTM|'
    r'CNN|attention|NLP|NER|POS|SVO|TF-IDF|BM25|FAISS|DeBERTa)\b',
    re.IGNORECASE,
)


# ── Utility ────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Basic normalisation: collapse whitespace, remove zero-width chars."""
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _lemmatize_phrase(tokens: list[Token]) -> str:
    """Return the lemmatized lower-case form of a token sequence."""
    return " ".join(t.lemma_.lower() for t in tokens if not t.is_punct)


# ── Stage sub-functions ────────────────────────────────────────────────────────

def extract_sentences(doc: Doc) -> list[str]:
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def extract_entities(doc: Doc) -> list[Entity]:
    seen = set()
    entities: list[Entity] = []

    for ent in doc.ents:
        if ent.label_ in DOMAIN_ENTITY_LABELS and ent.text.lower() not in seen:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
            ))
            seen.add(ent.text.lower())

    # Also scan for domain-specific technical terms not caught by spaCy NER
    for m in TECHNICAL_NER_PATTERN.finditer(doc.text):
        term = m.group(0)
        if term.lower() not in seen:
            entities.append(Entity(
                text=term,
                label="TECH_TERM",
                start_char=m.start(),
                end_char=m.end(),
            ))
            seen.add(term.lower())

    return entities


def extract_concepts(doc: Doc) -> list[str]:
    """
    Extract unique noun-phrase concepts.
    - Use spaCy's noun_chunks (already handles compound nouns)
    - Lemmatize and deduplicate
    - Filter very short or stopword-only chunks
    """
    seen: set[str] = set()
    concepts: list[str] = []

    for chunk in doc.noun_chunks:
        # Skip chunks that are only stopwords or punctuation
        content_tokens = [t for t in chunk if not t.is_stop and not t.is_punct and t.is_alpha]
        if not content_tokens:
            continue

        phrase = _lemmatize_phrase(chunk)
        if phrase and phrase not in seen and len(phrase) > 2:
            seen.add(phrase)
            concepts.append(phrase)

    return sorted(concepts)


def extract_claims(doc: Doc) -> list[Claim]:
    """
    Extract Subject-Verb-Object (SVO) triplets from dependency tree.

    Dependency path:
        nsubj → ROOT (verb) → dobj | attr | prep+pobj
    """
    claims: list[Claim] = []

    for sent in doc.sents:
        sent_text = sent.text.strip()

        for token in sent:
            # Find the root verb of each sentence
            if token.dep_ not in ("ROOT", "relcl", "advcl", "ccomp") or token.pos_ != "VERB":
                continue

            subject = _find_subject(token)
            obj = _find_object(token)

            if subject and obj:
                # Clean up predicate: use full verb phrase (aux + root)
                predicate = _build_predicate(token)
                claims.append(Claim(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    raw_sentence=sent_text,
                ))

    return claims


def _find_subject(verb: Token) -> Optional[str]:
    """Find the nominal subject of a verb token."""
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
            # Include full noun phrase if possible
            subtree_tokens = [t for t in child.subtree if not t.is_punct]
            return " ".join(t.text for t in subtree_tokens).strip()
    # Check parent (for relative clauses)
    if verb.head and verb.head.pos_ in ("NOUN", "PROPN"):
        return verb.head.lemma_
    return None


def _find_object(verb: Token) -> Optional[str]:
    """Find the direct object or attribute of a verb token."""
    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "acomp", "oprd"):
            subtree_tokens = [t for t in child.subtree if not t.is_punct]
            return " ".join(t.text for t in subtree_tokens).strip()
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    obj_tokens = [t for t in grandchild.subtree if not t.is_punct]
                    prep_text = child.text
                    obj_text = " ".join(t.text for t in obj_tokens)
                    return f"{prep_text} {obj_text}".strip()
    return None


def _build_predicate(verb: Token) -> str:
    """Build the full predicate string (aux verbs + root verb + negation)."""
    parts = []
    for child in verb.children:
        if child.dep_ in ("aux", "auxpass", "neg") and child.i < verb.i:
            parts.append(child.text)
    parts.append(verb.lemma_)
    return " ".join(parts)


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze(text: str) -> LinguisticOutput:
    """
    Main entry point for Stage 1.
    Returns a LinguisticOutput with sentences, entities, concepts, and claims.
    """
    text = _clean_text(text)
    if not text:
        return LinguisticOutput(sentences=[], entities=[], concepts=[], claims=[])

    nlp = _get_nlp()
    doc = nlp(text)

    sentences = extract_sentences(doc)
    entities = extract_entities(doc)
    concepts = extract_concepts(doc)
    claims = extract_claims(doc)

    return LinguisticOutput(
        sentences=sentences,
        entities=entities,
        concepts=concepts,
        claims=claims,
    )


# ── CLI / quick test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "BERT is a transformer-based model that uses bidirectional attention "
        "to pre-train language representations. Word2Vec generates word embeddings "
        "using a shallow neural network trained on large corpora."
    )
    result = analyze(sample)
    import json
    print(json.dumps(result.to_dict(), indent=2))
