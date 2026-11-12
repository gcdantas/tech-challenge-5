"""Linguagem compartilhada pelas transcrições e pelas notas clínicas sintéticas."""
from importlib.metadata import version
import json
import math
import os
import re

from .cache import ResponseCache, digest
from .config import AudioConfig, LanguageConfig


class AzureLanguage:
    mock = False
    name = "azure-ai-language"

    def __init__(self, config: LanguageConfig, cache: ResponseCache, *, allow_azure=False, client=None):
        self.config, self.cache = config, cache
        self.allow_azure, self.client = allow_azure, client
        self.version = version("azure-ai-textanalytics") if client is None else "sdk-simulado"

    def analyze(self, text: str) -> dict:
        if not text.strip() or len(text) > 5000:
            raise ValueError("Language requer texto não vazio com até 5000 caracteres por documento")
        parameters = {"text_sha256": digest(text.encode("utf-8")), "config": self.config.model_dump(),
                      "sdk": self.version, "endpoint": os.getenv("AZURE_LANGUAGE_ENDPOINT", ""),
                      "string_index_type": "UnicodeCodePoint"}
        return self.cache.get_or_call("azure-language", parameters, lambda: self._request(text))

    def _request(self, text: str) -> dict:
        if not self.allow_azure:
            raise PermissionError("Cache Language ausente. Use --allow-azure ou --mock.")
        if not os.getenv("AZURE_LANGUAGE_KEY") or not os.getenv("AZURE_LANGUAGE_ENDPOINT"):
            raise ValueError("Defina AZURE_LANGUAGE_KEY e AZURE_LANGUAGE_ENDPOINT no .env local")
        client = self.client
        owns_client = client is None
        if client is None:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential
            client = TextAnalyticsClient(os.environ["AZURE_LANGUAGE_ENDPOINT"],
                                         AzureKeyCredential(os.environ["AZURE_LANGUAGE_KEY"]),
                                         api_version=self.config.api_version,
                                         connection_timeout=10, read_timeout=60, retry_total=2)
        kwargs = {"language": self.config.language, "model_version": self.config.model_version,
                  "string_index_type": "UnicodeCodePoint", "disable_service_logs": True}
        try:
            resolved_versions = {}

            def capture(kind):
                def hook(response):
                    # O transporte síncrono azure-core expõe text(), não json().
                    body = json.loads(response.http_response.text())
                    resolved_versions[kind] = body.get("results", body).get("modelVersion")
                return hook

            sentiment = client.analyze_sentiment([text], raw_response_hook=capture("sentiment"), **kwargs)[0]
            entities = client.recognize_entities([text], raw_response_hook=capture("entities"), **kwargs)[0]
            if sentiment.is_error or entities.is_error:
                raise RuntimeError("Azure Language retornou erro de documento")
            sentences = [{"text": s.text, "offset": s.offset, "length": s.length,
                          "sentiment": s.sentiment,
                          "confidence_scores": {label: float(getattr(s.confidence_scores, label))
                                                for label in ("positive", "neutral", "negative")}}
                         for s in sentiment.sentences]
            result = {"sentiment": sentiment.sentiment, "sentences": sentences,
                      "entities": [{"text": e.text, "offset": e.offset, "length": e.length,
                                    "category": e.category, "confidence_score": e.confidence_score}
                                   for e in entities.entities],
                      "provider": self.name, "mock": False, "sdk_version": self.version,
                      "model_version": {"requested": self.config.model_version,
                                        "sentiment": resolved_versions.get("sentiment"),
                                        "entities": resolved_versions.get("entities")},
                      "warnings": [str(w.code) for w in sentiment.warnings + entities.warnings]}
            return result
        finally:
            if owns_client:
                client.close()


class MockLanguage:
    """Simulação determinística: não representa predições nem métricas do Azure."""
    mock = True
    name = "mock-language"
    version = "1"

    def analyze(self, text: str) -> dict:
        sentences = []
        for match in re.finditer(r"[^.!?\n]+[.!?]?", text):
            raw = match.group()
            content = raw.strip()
            if not content:
                continue
            offset = match.start() + len(raw) - len(raw.lstrip())
            negative = "muito angustiado" in content.casefold()
            sentences.append({"text": content, "offset": offset, "length": len(content),
                              "sentiment": "negative" if negative else "neutral",
                              "confidence_scores": {"positive": 0.01,
                                                    "neutral": 0.02 if negative else 0.98,
                                                    "negative": 0.97 if negative else 0.01}})
        return {"sentiment": "negative" if any(s["sentiment"] == "negative" for s in sentences) else "neutral",
                "sentences": sentences, "entities": [], "mock": True, "provider": self.name,
                "sdk_version": self.version, "model_version": {"sentiment": "mock-1", "entities": "mock-1"}}


def assertion(text: str, start: int, end: int) -> str:
    """Escopo conservador de negação/hipótese; não substitui NLP clínico."""
    # Uma conjunção adversativa encerra a negação; 'e/ou' mantém sua lista.
    prefix = re.split(r"[.!?;\n]|\b(?:mas|porém|entretanto|agora)\b", text[:start], flags=re.I)[-1]
    suffix = re.split(r"[.!?;\n,]", text[end:], maxsplit=1)[0]
    if re.search(r"\b(?:nega|negou|sem|não|ausência\s+de)\b", prefix, re.I):
        return "negated"
    if re.match(r"\s+(?:ausente|negad[oa]|não\s+(?:presente|observad[oa]))\b", suffix, re.I):
        return "negated"
    if re.search(r"\b(?:se|caso|risco\s+de|observar\s+se|investigar|suspeita\s+de)\b", prefix, re.I):
        return "uncertain"
    return "affirmed"


def annotate(text: str, response: dict, config: AudioConfig) -> dict:
    matches = []
    for term in config.terms:
        for match in re.finditer(term.pattern, text, re.I):
            methods = ["regex"]
            entities = [e for e in response.get("entities", [])
                        if e["confidence_score"] >= config.language.entity_threshold
                        and e["offset"] < match.end() and e["offset"] + e["length"] > match.start()]
            if entities:
                methods.append("azure-ner" if not response["mock"] else "mock-ner")
            matches.append({"term": match.group(), "canonical_term": term.term,
                            "offset": match.start(), "length": match.end() - match.start(),
                            "assertion": assertion(text, match.start(), match.end()),
                            "score": term.score, "methods": methods, "entities": entities})
    matches.sort(key=lambda item: item["offset"])
    return {**response, "critical_terms": [m for m in matches if m["assertion"] == "affirmed"],
            "excluded_mentions": [m for m in matches if m["assertion"] != "affirmed"]}


def negative_sentences(text: str, analysis: dict, config: AudioConfig) -> list[dict]:
    valid = []
    for sentence in analysis["sentences"]:
        offset, length = sentence["offset"], sentence["length"]
        scores = sentence["confidence_scores"]
        values = [scores.get(label, float("nan")) for label in ("positive", "neutral", "negative")]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in values):
            continue
        if (sentence["sentiment"] == "negative" and scores["negative"] >= config.language.negative_threshold
                and length >= config.language.min_sentence_chars
                and isinstance(offset, int) and isinstance(length, int) and offset >= 0
                and text[offset:offset + length] == sentence["text"]
                and abs(sum(values) - 1) < 0.02):
            valid.append(sentence)
    return valid
