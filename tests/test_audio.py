"""Testes I2 offline: SDKs simulados, sinais artificiais e contrato real."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace as NS
import wave

import numpy as np
import pytest

from contracts.finding import Finding, ModelInfo, read_findings
from contracts.patients import REPO_ROOT, load_patients
from modules.audio.cache import ResponseCache
from modules.audio.config import AudioConfig, VocalConfig, load_config
from modules.audio.language import AzureLanguage, MockLanguage, annotate, negative_sentences
from modules.audio.metrics import agreement, evaluate, term_metrics, wer
from modules.audio.pipeline import AudioPipeline, analyze
from modules.audio.run import main
from modules.audio.speech import AzureSpeech, MockSpeech, validate_segments, wav_info
from modules.audio.vocal import FEATURES, analyze_vocal, compare, extract_window, windows


def write_wav(path, duration=10, frequency=150, rate=16000, channels=1, samples=None):
    if samples is None:
        t = np.arange(round(duration * rate)) / rate
        samples = 0.2 * np.sin(2 * np.pi * frequency * t)
    with wave.open(str(path), "wb") as f:
        f.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        f.writeframes((np.asarray(samples) * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "cache")


@pytest.fixture
def azure_env(monkeypatch):
    for name, value in {"AZURE_SPEECH_KEY": "test-only-not-a-key", "AZURE_SPEECH_REGION": "test-region",
                        "AZURE_LANGUAGE_KEY": "test-only-not-a-key", "AZURE_LANGUAGE_ENDPOINT": "https://example.invalid"}.items():
        monkeypatch.setenv(name, value)


class Signal:
    def __init__(self):
        self.handlers = []

    def connect(self, callback):
        self.handlers.append(callback)

    def emit(self, event):
        for handler in self.handlers:
            handler(event)


class SpeechSDK:
    """Duplo do SDK com eventos e unidades reais; nenhuma rede."""
    OutputFormat = NS(Detailed="Detailed")
    ResultReason = NS(RecognizedSpeech="RecognizedSpeech")
    CancellationReason = NS(Error="Error", EndOfStream="EndOfStream")
    audio = NS(AudioConfig=lambda **kwargs: kwargs)

    class SpeechConfig:
        def __init__(self, **kwargs):
            self.words_requested = False

        def request_word_level_timestamps(self):
            self.words_requested = True

    def __init__(self, outcome="success"):
        self.outcome, self.starts, self.stops = outcome, 0, 0

    def SpeechRecognizer(self, speech_config, audio_config):
        assert speech_config.speech_recognition_language == "pt-BR"
        assert speech_config.words_requested
        recognizer = NS(recognized=Signal(), canceled=Signal(), session_stopped=Signal())

        def start():
            self.starts += 1
            if self.outcome == "timeout":
                return
            if self.outcome in ("error", "eof"):
                recognizer.canceled.emit(NS(reason="Error" if self.outcome == "error" else "EndOfStream"))
                return
            if self.outcome == "nomatch":
                recognizer.recognized.emit(NS(result=NS(reason="NoMatch")))
            else:
                for offset, text in [(10_000_000, "Estou confortável."), (60_000_000, "Sinto falta de ar.")]:
                    detail = {"NBest": [{"Confidence": 0.9, "Words": [
                        {"Word": text.split()[0], "Offset": offset, "Duration": 5_000_000}]}]}
                    recognizer.recognized.emit(NS(result=NS(reason="RecognizedSpeech", offset=offset,
                                                           duration=20_000_000, text=text, json=json.dumps(detail))))
            recognizer.session_stopped.emit(NS())

        def stop():
            self.stops += 1

        recognizer.start_continuous_recognition_async = lambda: NS(get=start)
        recognizer.stop_continuous_recognition_async = lambda: NS(get=stop)
        return recognizer


def test_speech_continuous_offsets_cache_and_config(tmp_path, cfg, cache, azure_env):
    path = write_wav(tmp_path / "audio.wav")
    sdk = SpeechSDK()
    speech = AzureSpeech(cfg.speech, cache, allow_azure=True, sdk=sdk)
    result = speech.transcribe(path)
    assert [(s["start"], s["end"]) for s in result["segments"]] == [(1, 3), (6, 8)]
    assert result["segments"][0]["words"][0]["end"] == 1.5
    speech.allow_azure = False
    assert speech.transcribe(path) == result
    assert sdk.starts == sdk.stops == 1
    speech.config = cfg.speech.model_copy(update={"language": "en-US"})
    with pytest.raises(PermissionError):
        speech.transcribe(path)


@pytest.mark.parametrize("outcome,error", [("error", RuntimeError), ("timeout", TimeoutError)])
def test_speech_failure_not_cached_and_stops(tmp_path, cfg, cache, azure_env, outcome, error):
    sdk = SpeechSDK(outcome)
    config = cfg.speech.model_copy(update={"timeout_seconds": 0.01})
    speech = AzureSpeech(config, cache, allow_azure=True, sdk=sdk)
    with pytest.raises(error):
        speech.transcribe(write_wav(tmp_path / "a.wav"))
    assert sdk.stops == 1
    assert not list(cache.directory.rglob("*.json"))


@pytest.mark.parametrize("outcome", ["nomatch", "eof"])
def test_speech_silence_does_not_invent_segments(tmp_path, cfg, cache, azure_env, outcome):
    speech = AzureSpeech(cfg.speech, cache, allow_azure=True, sdk=SpeechSDK(outcome))
    assert speech.transcribe(write_wav(tmp_path / "a.wav"))["segments"] == []


def test_missing_credentials_no_implicit_mock(tmp_path, cfg, cache, monkeypatch):
    for name in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="AZURE_SPEECH_KEY"):
        AzureSpeech(cfg.speech, cache, allow_azure=True, sdk=SpeechSDK()).transcribe(write_wav(tmp_path / "a.wav"))
    with pytest.raises(ValueError, match="AZURE_LANGUAGE_KEY"):
        AzureLanguage(cfg.language, cache, allow_azure=True, client=NS()).analyze("Sinto falta de ar.")


def test_network_requires_explicit_authorization(tmp_path, cfg, cache, azure_env):
    with pytest.raises(PermissionError):
        AzureSpeech(cfg.speech, cache, sdk=SpeechSDK()).transcribe(write_wav(tmp_path / "a.wav"))
    with pytest.raises(PermissionError):
        AzureLanguage(cfg.language, cache, client=NS()).analyze("Sinto falta de ar.")


@pytest.mark.parametrize("rate,channels", [(8000, 1), (16000, 2)])
def test_wav_format_rejected(tmp_path, rate, channels):
    with pytest.raises(ValueError, match="WAV PCM"):
        wav_info(write_wav(tmp_path / "bad.wav", rate=rate, channels=channels))


def test_absent_empty_and_truncated_audio(tmp_path):
    with pytest.raises(FileNotFoundError):
        wav_info(tmp_path / "missing.wav")
    with pytest.raises(ValueError, match="vazio"):
        wav_info(write_wav(tmp_path / "empty.wav", duration=0))
    p = write_wav(tmp_path / "broken.wav")
    p.write_bytes(p.read_bytes()[:-4])
    with pytest.raises(ValueError, match="truncado"):
        wav_info(p)


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 1), (0, float("nan")), (0, float("inf"))])
def test_invalid_segments_rejected(start, end):
    with pytest.raises(ValueError):
        validate_segments([{"start": start, "end": end, "text": "teste"}])


def test_mock_requires_explicit_marker(tmp_path):
    p = tmp_path / "fixture.json"
    p.write_text('{"segments": []}')
    with pytest.raises(ValueError, match="mock=true"):
        MockSpeech().transcribe(p)


class LanguageClient:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def analyze_sentiment(self, documents, **kwargs):
        # Classes reais do SDK impedem mocks que inventem campos inexistentes.
        from azure.ai.textanalytics import AnalyzeSentimentResult, SentenceSentiment, SentimentConfidenceScores
        self.calls += 1
        assert kwargs["language"] == "pt"
        assert kwargs["string_index_type"] == "UnicodeCodePoint"
        assert kwargs["disable_service_logs"] is True
        kwargs["raw_response_hook"](NS(http_response=NS(text=lambda: '{"results":{"modelVersion":"test-sentiment"}}')))
        text = documents[0]
        return [AnalyzeSentimentResult(id="0", sentiment="negative", sentences=[
            SentenceSentiment(text=text, offset=0, length=len(text), sentiment="negative",
                              confidence_scores=SentimentConfidenceScores(positive=0.01, neutral=0.02, negative=0.97))])]

    def recognize_entities(self, documents, **kwargs):
        from azure.ai.textanalytics import RecognizeEntitiesResult, CategorizedEntity
        self.calls += 1
        if self.fail:
            return [NS(is_error=True)]
        text = documents[0]
        return [RecognizeEntitiesResult(id="0", entities=[
            CategorizedEntity(text="falta de ar", category="Skill", offset=text.index("falta de ar"),
                              length=11, confidence_score=0.99)])]


def test_language_sdk_sentence_sentiment_entities_and_cache(cfg, cache, azure_env):
    client = LanguageClient()
    language = AzureLanguage(cfg.language, cache, allow_azure=True, client=client)
    text = "😟 Sinto falta de ar."
    result = language.analyze(text)
    assert result["model_version"]["sentiment"] == "test-sentiment"
    assert result["model_version"]["entities"] is None  # SDK não inventa versão ausente.
    annotated = annotate(text, result, cfg)
    assert annotated["critical_terms"][0]["methods"] == ["regex", "azure-ner"]
    assert text[annotated["critical_terms"][0]["offset"]:].startswith("falta de ar")
    assert len(negative_sentences(text, annotated, cfg)) == 1
    language.allow_azure = False
    assert language.analyze(text) == result
    assert client.calls == 2
    assert "test-only-not-a-key" not in "".join(p.read_text() for p in cache.directory.rglob("*.json"))


def test_language_error_not_cached(cfg, cache, azure_env):
    with pytest.raises(RuntimeError, match="erro de documento"):
        AzureLanguage(cfg.language, cache, allow_azure=True, client=LanguageClient(True)).analyze("falta de ar")
    assert not list(cache.directory.rglob("*.json"))


@pytest.mark.parametrize("text,expected", [
    ("Tenho falta de ar e dor no peito. Ontem desmaiei. Apresento dispnéia.", {"falta de ar", "dor no peito", "desmaiei", "dispneia"}),
    ("Nega falta de ar, dor no peito e desmaio. Sem dispneia ou sonolência.", set()),
    ("Sem dor no peito, mas sinto falta de ar.", {"falta de ar"}),
    ("Se tiver falta de ar, avisar a equipe.", set()),
    ("Dispneia ausente. Dor no peito negada.", set()),
    ("Hoje está confortável e alerta.", set()),
])
def test_critical_terms_and_assertion(cfg, text, expected):
    result = annotate(text, MockLanguage().analyze(text), cfg)
    assert {term["canonical_term"] for term in result["critical_terms"]} == expected


def test_sentiment_by_sentence_valid_evidence(cfg):
    text = "Estou confortável. Estou muito angustiado com essa mudança."
    result = MockLanguage().analyze(text)
    sentences = negative_sentences(text, result, cfg)
    assert len(result["sentences"]) == 2 and len(sentences) == 1
    broken = deepcopy(result)
    broken["sentences"][1]["offset"] = 0
    assert negative_sentences(text, broken, cfg) == []
    broken = deepcopy(result)
    broken["sentences"][1]["confidence_scores"]["negative"] = float("nan")
    assert negative_sentences(text, broken, cfg) == []


def test_cache_content_parameters_failures(tmp_path):
    cache = ResponseCache(tmp_path)
    calls = []
    def call():
        calls.append(1)
        return {"answer": len(calls)}
    assert cache.get_or_call("service", {"x": 1}, call) == {"answer": 1}
    assert cache.get_or_call("service", {"x": 1}, call) == {"answer": 1}
    assert cache.get_or_call("service", {"x": 2}, call) == {"answer": 2}
    assert len(calls) == 2
    with pytest.raises(RuntimeError):
        cache.get_or_call("failed", {}, lambda: (_ for _ in ()).throw(RuntimeError("failed")))
    assert not (tmp_path / "failed").exists()


def test_notes_comparison_order_negations_and_contract(tmp_path):
    pipeline = AudioPipeline(mock=True, outputs=tmp_path)
    result = pipeline.notes(["data/notes/P001_nota_2.txt", "data/notes/P001_nota_1.txt"], "P001")
    first, recent = result.document["notes"]
    assert first["critical_terms"] == []
    assert first["timestamp_start"] == "2026-11-02T10:00:00-03:00"
    assert recent["timestamp_start"] == "2026-11-03T10:40:00-03:00"
    assert set(recent["comparison"]["added_terms"]) == {"dispneia", "sonolência", "falta de ar"}
    assert all(f.source == "text" and f.evidence["ausente_na_anterior"] for f in result.findings)
    assert all(f.evidence["synthetic"] for f in result.findings)
    for f in result.findings:
        assert Finding.model_validate_json(f.to_json_line()) == f


def test_scenario_four_outputs_loaders_and_no_fake_vocal(tmp_path):
    from app.loaders import load_findings, load_transcript
    for module in ("video", "anomaly"):
        (tmp_path / module / "P001").mkdir(parents=True)
        (tmp_path / module / "P001/real.bin").write_bytes(b"untouched")
    pipeline = AudioPipeline(mock=True, outputs=tmp_path)
    summary = pipeline.scenario()
    assert summary["audio_findings"] == 2 and summary["text_findings"] == 3
    assert summary["vocal_status"] == "not_evaluated"
    loaded = load_findings(tmp_path, "P001")
    assert not loaded.errors and len(loaded.findings) == 5
    transcript = load_transcript(tmp_path, "P001")
    assert transcript["mock"] is True
    assert transcript["segments"][2]["timestamp_start"] == "2026-11-03T10:31:15-03:00"
    assert not any(f.type == "indicador_vocal_alterado" for f in loaded.findings)
    assert all(f.model.name.startswith("mock-") and f.description.startswith("[MOCK]") for f in loaded.findings)
    for source, name in [("audio", "transcript.json"), ("text", "notes.json")]:
        assert (tmp_path / source / "P001/findings.jsonl").exists()
        assert (tmp_path / source / "P001" / name).exists()
    assert all((tmp_path / mod / "P001/real.bin").read_bytes() == b"untouched" for mod in ("video", "anomaly"))


def test_fusion_accepts_real_contract_audio_without_changing_rule(tmp_path):
    from app.fusion import fuse, load_fusion_config
    pf = load_patients()
    result = AudioPipeline(mock=True, outputs=tmp_path).audio("modules/audio/fixtures/P001_mock.json", "P001", artifact_name="consulta")
    # Só no teste: complementos simulados, nunca escritos como detecção real.
    extras = [Finding(patient_id="P001", source=source, type=kind,
                      timestamp_start=pf.demo_time(offset), timestamp_end=pf.demo_time(offset + 5),
                      score=0.8, severity="medium", description="[MOCK] teste de integração",
                      evidence={"mock": True}, model=ModelInfo(name="mock-test", version="1"))
              for source, kind, offset in [("audio", "indicador_vocal_alterado", 1830), ("vitals", "dessaturacao", 2700)]]
    alerts = fuse(result.findings + extras, load_fusion_config())
    assert any(a.rule == "respiratorio_vocal_spo2" for a in alerts)


def test_pipeline_sdk_timestamps(tmp_path, cfg, cache, azure_env):
    path = write_wav(tmp_path / "consult.wav")
    pipeline = AudioPipeline(outputs=tmp_path, speech=AzureSpeech(cfg.speech, cache, allow_azure=True, sdk=SpeechSDK()),
                             language=MockLanguage())
    result = pipeline.audio(path, "P001", artifact_name="consulta")
    assert result.document["segments"][0]["timestamp_start"] == "2026-11-03T10:30:01-03:00"
    assert all(f.timestamp_start.utcoffset() == timedelta(hours=-3) for f in result.findings)


def test_mock_refuses_overwrite_of_real_outputs(tmp_path):
    target = tmp_path / "audio/P001"
    target.mkdir(parents=True)
    (target / "findings.jsonl").write_text("")
    (target / "transcript.json").write_text('{"mock": false}')
    with pytest.raises(FileExistsError, match="saída real"):
        AudioPipeline(mock=True, outputs=tmp_path).scenario()
    assert (target / "transcript.json").read_text() == '{"mock": false}'


def test_missing_files_and_unmapped_artifacts(tmp_path):
    p = AudioPipeline(mock=True, outputs=tmp_path)
    with pytest.raises(FileNotFoundError):
        p.audio(tmp_path / "none.wav", "P001")
    with pytest.raises(FileNotFoundError):
        p.notes([tmp_path / "none.txt"], "P001")
    file = tmp_path / "unmapped.txt"
    file.write_text("nota")
    with pytest.raises(ValueError, match="não mapeada"):
        p.notes([file], "P001")
    with pytest.raises(KeyError):
        p.scenario("P999")


def test_cli(tmp_path, capsys):
    assert main(["--scenario", "--patient", "P001", "--mock", "--outputs", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["mock"] is True
    assert main(["--input", "missing.wav", "--patient", "P001", "--out", str(tmp_path / "fail.jsonl")]) == 2
    assert "não encontrada" in capsys.readouterr().err
    assert main(["--input", "modules/audio/fixtures/P001_mock.json", "--artifact", "consulta", "--patient", "P001",
                 "--mock", "--out", str(tmp_path / "single/findings.jsonl")]) == 0
    assert (tmp_path / "single/transcript.json").exists()


def test_public_interface_uses_cached_real_path(monkeypatch):
    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["allow_azure"] is False
        def audio(self, path, patient_id):
            assert patient_id == "P001"
            return NS(findings=[])
    monkeypatch.setenv("TC_AUDIO_ALLOW_AZURE", "0")
    monkeypatch.setattr("modules.audio.pipeline.AudioPipeline", Pipeline)
    assert analyze("data/audio/P001_consulta.wav", "P001") == []


def test_config_rejects_invalid_thresholds(cfg):
    raw = cfg.model_dump()
    raw["severity"]["high"] = 0.1
    with pytest.raises(ValueError):
        AudioConfig.model_validate(raw)


def test_vocal_zscore_and_edge_cases():
    cfg = VocalConfig()
    base = [{name: float(i) for name in FEATURES} for i in range(5)]
    current = {name: 10.0 for name in FEATURES}
    result = compare(current, base, cfg)
    assert result["altered"] and 0 <= result["score"] <= 1
    assert result["zscores"]["pitch"] == pytest.approx(8 / np.std(range(5), ddof=1))
    assert not compare(current, base[:1], cfg)["altered"]
    zero = [{name: 1.0 for name in FEATURES}] * 6
    assert not compare(current, zero, cfg)["altered"]
    assert set(compare(current, zero, cfg)["skipped"].values()) == {"desvio_padrao_zero_ou_insuficiente"}
    assert not compare({name: float("nan") for name in FEATURES}, base, cfg)["altered"]
    missing = [{name: None for name in FEATURES}] * 6
    assert compare(current, missing, cfg)["zscores"] == {}


def test_vocal_features_measured_from_synthetic_signal():
    cfg = VocalConfig()
    t = np.arange(5 * 16000) / 16000
    signal = 0.2 * np.sin(2 * np.pi * 150 * t)
    signal[16000:24000] = 0
    words = [{"text": "a", "start": i, "end": i + 0.2} for i in (0, 2, 3, 4)]
    result = extract_window(signal, words, 0, cfg)
    assert result["taxa_fala"] == 48
    assert result["pausas"] == pytest.approx(0.1, abs=0.01)
    assert result["pitch"] == pytest.approx(150, abs=1)
    assert result["jitter"] is not None and result["shimmer"] is not None
    assert extract_window(signal, None, 0, cfg)["taxa_fala"] is None
    assert all(v is None for v in extract_window(np.zeros_like(signal), words, 0, cfg).values())
    assert all(v is None for v in extract_window(signal[:100], words, 0, cfg).values())


def metadata(path, speaker="actor-a", **extra):
    path.with_suffix(".wav.json").write_text(json.dumps({"patient_id": "P001", "speaker_id": speaker,
                                                        "single_speaker": True, **extra}))


def test_vocal_same_speaker_and_baseline_guards(tmp_path):
    path = write_wav(tmp_path / "a.wav", duration=35)
    baseline = write_wav(tmp_path / "base.wav", duration=30)
    cfg = VocalConfig()
    assert analyze_vocal(path, [], "P001", cfg)["status"] == "not_evaluated"
    metadata(path)
    assert analyze_vocal(path, [], "P001", cfg)["reason"] == "neutral_prefix_not_confirmed"
    metadata(path, neutral_prefix_seconds=30)
    metadata(baseline, speaker="actor-b", condition="neutral")
    with pytest.raises(ValueError, match="mesmo locutor"):
        analyze_vocal(path, [], "P001", cfg, baseline)
    metadata(baseline, condition="neutral")
    # Tom puro tem baseline sem variância: nenhuma alteração artificial.
    result = analyze_vocal(path, [], "P001", cfg, baseline)
    assert result["status"] == "insufficient_data"
    assert not any(w["altered"] for w in result["windows"])
    result = analyze_vocal(path, [], "P001", cfg)
    assert all(w["start"] >= 30 for w in result["windows"])


def test_metrics_only_with_reference():
    assert wer("eu sinto falta de ar", "eu tenho falta de ar")["wer"] == 0.2
    assert wer("Olá, mundo!", "olá mundo")["wer"] == 0
    with pytest.raises(ValueError):
        wer("", "palavra")
    assert term_metrics(["falta de ar"], ["falta de ar", "dispneia"])["precision"] == 0.5
    assert term_metrics([], [])["recall"] is None
    assert agreement([True, False, True], [True, True, None]) == {
        "agreement": 0.5, "evaluated": 2, "total": 3, "coverage": 2 / 3}
    with pytest.raises(ValueError):
        evaluate({"human_reviewed": False})


def test_measured_vocal_finding_pipeline_and_timestamps(tmp_path):
    # 6 janelas neutras com variação controlada e uma janela claramente diferente.
    chunks = []
    t = np.arange(5 * 16000) / 16000
    for frequency, pause in [(140, 0.1), (145, 0.2), (150, 0.3), (155, 0.4), (160, 0.5), (165, 0.6), (280, 1.6)]:
        chunk = 0.2 * np.sin(2 * np.pi * frequency * t)
        chunk[16000:16000 + round(pause * 16000)] = 0
        chunks.append(chunk)
    path = write_wav(tmp_path / "signal.wav", samples=np.concatenate(chunks))
    metadata(path, neutral_prefix_seconds=30)
    # Não simula voz clínica nem Azure: apenas sinal de teste e ausência de texto.
    speech = NS(mock=False, transcribe=lambda p: {"segments": [], "mock": False, "provider": "test-only"})
    pipeline = AudioPipeline(outputs=tmp_path, speech=speech, language=NS(mock=False))
    result = pipeline.audio(path, "P001", artifact_name="consulta")
    assert result.document["vocal"]["status"] == "evaluated"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.type == "indicador_vocal_alterado" and finding.source == "audio"
    assert finding.timestamp_start.isoformat() == "2026-11-03T10:30:30-03:00"
    assert finding.timestamp_end.isoformat() == "2026-11-03T10:30:35-03:00"
    assert {"pitch", "pausas"} <= set(finding.evidence["changed_features"])
    assert Finding.model_validate_json(finding.to_json_line()) == finding


def test_notes_changes_include_removed_and_persisting(tmp_path):
    import yaml
    previous, recent = tmp_path / "previous.txt", tmp_path / "recent.txt"
    previous.write_text("Refere falta de ar e dor no peito.", encoding="utf-8")
    recent.write_text("Refere falta de ar. Nega dor no peito.", encoding="utf-8")
    data = yaml.safe_load((REPO_ROOT / "data/patients.yaml").read_text(encoding="utf-8"))
    data["patients"]["P001"]["artifacts"]["nota_anterior"]["path"] = str(previous)
    data["patients"]["P001"]["artifacts"]["nota_recente"]["path"] = str(recent)
    patients = tmp_path / "patients.yaml"
    patients.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = AudioPipeline(mock=True, outputs=tmp_path, patients_path=patients).notes([recent, previous], "P001")
    changes = result.document["notes"][1]["comparison"]
    assert changes["removed_terms"] == ["dor no peito"]
    assert changes["persisting_terms"] == ["falta de ar"]
    assert changes["added_terms"] == []
    assert result.findings[-1].evidence["ausente_na_anterior"] is False


def test_speech_cache_changes_with_file_contents(tmp_path, cfg, cache, azure_env):
    path = write_wav(tmp_path / "a.wav")
    sdk = SpeechSDK()
    speech = AzureSpeech(cfg.speech, cache, allow_azure=True, sdk=sdk)
    speech.transcribe(path)
    write_wav(path, frequency=180)
    speech.transcribe(path)
    assert sdk.starts == 2


def test_baseline_identity_rejected_before_speech_call(tmp_path):
    path, baseline = write_wav(tmp_path / "a.wav"), write_wav(tmp_path / "base.wav")
    metadata(path)
    metadata(baseline, speaker="other", condition="neutral")
    pipeline = AudioPipeline(outputs=tmp_path)
    with pytest.raises(ValueError, match="mesmo locutor"):
        pipeline.audio(path, "P001", artifact_name="consulta", baseline=baseline)
    with pytest.raises(ValueError, match="próprio arquivo"):
        pipeline.audio(path, "P001", artifact_name="consulta", baseline=path)


def test_language_real_sdk_with_offline_http_transport(cfg, cache, azure_env):
    """Exercita serialização/deserialização do SDK real sem enviar HTTP à Azure."""
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.pipeline.transport import HttpResponse, HttpTransport

    class Response(HttpResponse):
        def __init__(self, request, payload):
            super().__init__(request, None)
            self.status_code = 200
            self.headers = {"content-type": "application/json"}
            self.reason = "OK"
            self.content_type = "application/json"
            self.payload = json.dumps(payload).encode("utf-8")

        def body(self):
            return self.payload

    class Transport(HttpTransport):
        def __init__(self):
            self.calls = []

        def open(self):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def send(self, request, **kwargs):
            body = json.loads(request.body)
            self.calls.append(body)
            assert body["parameters"]["loggingOptOut"] is True
            assert body["parameters"]["stringIndexType"] == "UnicodeCodePoint"
            document = body["analysisInput"]["documents"][0]
            assert document["language"] == "pt"
            text = document["text"]
            if body["kind"] == "SentimentAnalysis":
                scores = {"negative": 0.97, "neutral": 0.02, "positive": 0.01}
                result = {"id": document["id"], "sentiment": "negative", "confidenceScores": scores,
                          "warnings": [], "sentences": [{"text": text, "offset": 0, "length": len(text),
                                                          "sentiment": "negative", "confidenceScores": scores}]}
            else:
                assert body["kind"] == "EntityRecognition"
                result = {"id": document["id"], "warnings": [], "entities": [
                    {"text": "falta de ar", "category": "Skill", "offset": text.index("falta de ar"),
                     "length": 11, "confidenceScore": 0.99}]}
            return Response(request, {"kind": body["kind"] + "Results",
                                      "results": {"documents": [result], "errors": [], "modelVersion": "offline-fixture"}})

    transport = Transport()
    with TextAnalyticsClient("https://example.invalid", AzureKeyCredential("test-only"),
                             api_version=cfg.language.api_version, transport=transport) as client:
        result = AzureLanguage(cfg.language, cache, allow_azure=True, client=client).analyze("Sinto falta de ar.")
    assert len(transport.calls) == 2
    assert result["sentences"][0]["confidence_scores"]["negative"] == 0.97
    assert result["model_version"]["sentiment"] == "offline-fixture"
    assert result["model_version"]["entities"] == "offline-fixture"
