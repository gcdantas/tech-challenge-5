"""Interface do I2 e montagem dos artefatos consumidos pela aplicação existente."""
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

from contracts.finding import Finding, ModelInfo, read_findings, write_findings
from contracts.patients import DEFAULT_PATIENTS_PATH, REPO_ROOT, load_patients
from .cache import ResponseCache, write_json
from .config import load_config
from .language import AzureLanguage, MockLanguage, annotate, negative_sentences
from .speech import AzureSpeech, MockSpeech
from .vocal import analyze_vocal, speaker_metadata


@dataclass
class AnalysisResult:
    findings: list[Finding]
    document: dict


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


class AudioPipeline:
    def __init__(self, *, mock=False, allow_azure=False, config_path=None,
                 patients_path=None, outputs=None, speech=None, language=None):
        load_dotenv(REPO_ROOT / ".env", override=False)
        self.config = load_config(config_path)
        self.patients = load_patients(patients_path or os.getenv("TC_PATIENTS_PATH") or DEFAULT_PATIENTS_PATH)
        self.outputs = resolve(outputs or os.getenv("TC_OUTPUTS_DIR") or "outputs")
        cache = ResponseCache(self.outputs / "audio/cache")
        if mock and allow_azure:
            raise ValueError("--mock e --allow-azure são mutuamente exclusivos")
        self.speech = speech or (MockSpeech() if mock else AzureSpeech(self.config.speech, cache, allow_azure=allow_azure))
        self.language = language or (MockLanguage() if mock else AzureLanguage(self.config.language, cache, allow_azure=allow_azure))
        self.mock = bool(mock or self.speech.mock or self.language.mock)

    def _artifact(self, path: Path, patient_id: str, kind: str, artifact_name=None):
        if artifact_name:
            art = self.patients.artifact(patient_id, artifact_name)
            if art.kind != kind:
                raise ValueError(f"Artefato {artifact_name} não é {kind}")
            return artifact_name
        found = self.patients.find_artifact_by_path(patient_id, path)
        if found is None or found[1].kind != kind:
            raise ValueError("Entrada não mapeada em patients.yaml; informe --artifact para usar seu offset")
        return found[0]

    def _finding(self, patient_id, source, type_, start, end, score, description, evidence, artifacts, model=None):
        return Finding(patient_id=patient_id, source=source, type=type_, timestamp_start=start,
                       timestamp_end=end, score=score, severity=self.config.severity.level(score),
                       description=("[MOCK] " if self.mock else "") + description,
                       evidence={**evidence, "mock": self.mock},
                       model=model or ModelInfo(name=("mock-language-regex" if self.mock else "azure-language-regex"),
                                                version="1"), artifacts=[str(p) for p in artifacts])

    def _language_findings(self, patient_id, source, text, analysis, start, end, path, extra=None):
        result = []
        provenance = {"provider": analysis["provider"], "sdk_version": analysis["sdk_version"],
                      "language_model_version": analysis["model_version"], "rule_version": "1",
                      "segment_text": text, **(extra or {})}
        for term in analysis["critical_terms"]:
            result.append(self._finding(patient_id, source, "termo_critico", start, end, term["score"],
                                       f'Menção afirmada do termo crítico "{term["term"]}"',
                                       {**provenance, **term}, [path]))
        for sentence in negative_sentences(text, analysis, self.config):
            result.append(self._finding(patient_id, source, "sentimento_muito_negativo", start, end,
                                       sentence["confidence_scores"]["negative"],
                                       "Sentimento textual muito negativo na frase; não é diagnóstico",
                                       {**provenance, "sentence": sentence, "timing_precision": "segment"}, [path]))
        return result

    def audio(self, input: str | Path, patient_id: str, *, artifact_name=None, baseline=None) -> AnalysisResult:
        path = resolve(input)
        if not path.is_file():
            raise FileNotFoundError(f"Entrada não encontrada: {path}")
        artifact = self._artifact(path, patient_id, "audio", artifact_name)
        baseline_path = resolve(baseline) if baseline else None
        # Erros de identidade são detectados antes de qualquer chamada paga.
        if baseline_path:
            if self.mock:
                raise ValueError("Baseline acústico não se aplica à transcrição mock")
            if baseline_path.resolve() == path.resolve():
                raise ValueError("Baseline separado não pode ser o próprio arquivo da consulta")
            current, other = speaker_metadata(path, patient_id), speaker_metadata(baseline_path, patient_id)
            if not current or not other or current["speaker_id"] != other["speaker_id"] or other.get("condition") != "neutral":
                raise ValueError("Baseline neutro e consulta devem pertencer ao mesmo locutor confirmado")
        response = self.speech.transcribe(path)
        findings, segments = [], []
        for segment in response["segments"]:
            analysis = annotate(segment["text"], self.language.analyze(segment["text"]), self.config)
            start = self.patients.artifact_time(patient_id, artifact, segment["start"])
            end = self.patients.artifact_time(patient_id, artifact, segment["end"])
            segments.append({**segment, "timestamp_start": start.isoformat(), "timestamp_end": end.isoformat(),
                             "sentiment": analysis["sentiment"], "sentences": analysis["sentences"],
                             "critical_terms": analysis["critical_terms"], "excluded_mentions": analysis["excluded_mentions"],
                             "entities": analysis["entities"], "language_analysis": analysis})
            findings.extend(self._language_findings(patient_id, "audio", segment["text"], analysis, start, end,
                                                   path, {"media_start_s": segment["start"], "media_end_s": segment["end"]}))
        if self.mock:
            vocal = {"status": "not_evaluated", "reason": "mock_has_no_acoustic_evidence", "windows": []}
        else:
            base_segments = self.speech.transcribe(baseline_path)["segments"] if baseline_path and self.config.vocal.enabled else None
            vocal = analyze_vocal(path, response["segments"], patient_id, self.config.vocal, baseline_path, base_segments)
        for window in vocal["windows"]:
            if not window["altered"]:
                continue
            from importlib.metadata import version
            findings.append(self._finding(patient_id, "audio", "indicador_vocal_alterado",
                                         self.patients.artifact_time(patient_id, artifact, window["start"]),
                                         self.patients.artifact_time(patient_id, artifact, window["end"]),
                                         window["score"],
                                         "Indicador experimental de alteração vocal, baseado em características acústicas",
                                         {**window, "baseline": vocal["baseline"], "speaker_id": vocal["speaker_id"],
                                          "units": vocal["units"], "threshold": self.config.vocal.z_threshold,
                                          "score_z_cap": self.config.vocal.score_z_cap},
                                         [path] + ([baseline_path] if baseline_path else []),
                                         ModelInfo(name="praat-parselmouth-zscore", version=version("praat-parselmouth"))))
        return AnalysisResult(findings, {"mock": self.mock, "patient_id": patient_id, "audio": str(path),
                                        "language": self.config.speech.language,
                                        "aviso": "Transcrição FICTÍCIA; Azure não utilizado." if self.mock else "Protótipo experimental.",
                                        "speech": {k: v for k, v in response.items() if k != "segments"},
                                        "segments": segments, "vocal": vocal})

    def notes(self, inputs: list[str | Path], patient_id: str, *, artifact_name=None) -> AnalysisResult:
        entries = []
        if not inputs:
            raise ValueError("Nenhuma nota fornecida")
        for input in inputs:
            path = resolve(input)
            if not path.is_file():
                raise FileNotFoundError(f"Nota não encontrada: {path}")
            artifact = self._artifact(path, patient_id, "note", artifact_name if len(inputs) == 1 else None)
            text = path.read_text(encoding="utf-8-sig").strip()
            if not text or len(text) > 5000:
                raise ValueError("Notas devem conter entre 1 e 5000 caracteres")
            entries.append((self.patients.artifact_time(patient_id, artifact), artifact, path, text))
        entries.sort(key=lambda row: row[0])
        if len({entry[1] for entry in entries}) != len(entries):
            raise ValueError("Notas duplicadas para o mesmo artefato")
        findings, notes, previous_terms, previous = [], [], set(), None
        for timestamp, artifact, path, text in entries:
            analysis = annotate(text, self.language.analyze(text), self.config)
            terms = {term["canonical_term"] for term in analysis["critical_terms"]}
            changes = {"previous_note": previous, "added_terms": sorted(terms - previous_terms) if previous else [],
                       "removed_terms": sorted(previous_terms - terms) if previous else [],
                       "persisting_terms": sorted(terms & previous_terms) if previous else []}
            extra = {"note": artifact, "synthetic": text.startswith("NOTA SINTÉTICA"), "comparison": changes}
            note_findings = self._language_findings(patient_id, "text", text, analysis, timestamp, timestamp, path, extra)
            # Mudanças ficam no Finding existente, sem criar eventos duplicados para a fusão.
            for finding in note_findings:
                if finding.type == "termo_critico":
                    finding.evidence["ausente_na_anterior"] = (finding.evidence["canonical_term"] not in previous_terms
                                                               if previous else None)
            findings.extend(note_findings)
            notes.append({"note": artifact, "path": str(path), "text": text, "synthetic": extra["synthetic"],
                          "timestamp_start": timestamp.isoformat(), "timestamp_end": timestamp.isoformat(),
                          "sentiment": analysis["sentiment"], "sentences": analysis["sentences"],
                          "critical_terms": analysis["critical_terms"], "excluded_mentions": analysis["excluded_mentions"],
                          "comparison": changes, "language_analysis": analysis})
            previous, previous_terms = artifact, terms
        return AnalysisResult(findings, {"mock": self.mock, "patient_id": patient_id,
                                        "aviso": "As notas marcadas synthetic são fictícias, não são registros clínicos reais.",
                                        "notes": notes})

    def save(self, result: AnalysisResult, findings_path: str | Path, document_name: str) -> None:
        path = resolve(findings_path)
        document_path = path.parent / document_name
        # Uma execução de demonstração jamais substitui arquivos reais.
        if self.mock:
            for existing in (path, document_path):
                if existing.exists():
                    if existing.suffix == ".jsonl":
                        records = read_findings(existing)
                        safe = bool(records) and all(f.evidence.get("mock") is True for f in records)
                        # JSONL vazio só é mock quando seu documento acompanhante o confirma.
                        if not records and document_path.exists():
                            import json
                            safe = json.loads(document_path.read_text(encoding="utf-8")).get("mock") is True
                    else:
                        import json
                        safe = json.loads(existing.read_text(encoding="utf-8")).get("mock") is True
                    if not safe:
                        raise FileExistsError(f"Mock não pode sobrescrever saída real: {existing}")
        write_findings(path, result.findings)
        read_findings(path)  # Verificação pelo contrato comum, inclusive após serialização.
        write_json(document_path, result.document)

    def scenario(self, patient_id="P001", *, input=None, baseline=None) -> dict:
        artifacts = self.patients.patient(patient_id).artifacts
        audio_names = [name for name, art in artifacts.items() if art.kind == "audio"]
        if len(audio_names) != 1:
            raise ValueError("Cenário requer exatamente uma consulta mapeada")
        name = audio_names[0]
        if self.mock and patient_id != "P001" and input is None:
            raise ValueError("Fixture padrão pertence apenas ao P001")
        audio_input = input or (REPO_ROOT / "modules/audio/fixtures/P001_mock.json" if self.mock else artifacts[name].path)
        note_paths = [art.path for art in artifacts.values() if art.kind == "note"]
        if not audio_input or not note_paths or any(p is None for p in note_paths):
            raise ValueError("Cenário sem caminhos de consulta/notas")
        # Verifica entradas das duas modalidades antes de consumir Azure.
        for path in [audio_input, *note_paths]:
            if not resolve(path).is_file():
                raise FileNotFoundError(f"Entrada não encontrada: {path}")
        audio = self.audio(audio_input, patient_id, artifact_name=name, baseline=baseline)
        notes = self.notes(note_paths, patient_id)
        self.save(audio, self.outputs / "audio" / patient_id / "findings.jsonl", "transcript.json")
        self.save(notes, self.outputs / "text" / patient_id / "findings.jsonl", "notes.json")
        return {"audio_findings": len(audio.findings), "text_findings": len(notes.findings), "mock": self.mock,
                "vocal_status": audio.document["vocal"]["status"], "outputs": str(self.outputs)}


def analyze(input, patient_id) -> list[Finding]:
    """Interface comum. Azure somente com TC_AUDIO_ALLOW_AZURE=1; nunca mock implícito."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    pipeline = AudioPipeline(allow_azure=os.getenv("TC_AUDIO_ALLOW_AZURE") == "1")
    path = resolve(input)
    if path.is_dir():
        return pipeline.notes(sorted(path.glob("*.txt")), patient_id).findings
    if path.suffix.lower() == ".txt":
        return pipeline.notes([path], patient_id).findings
    return pipeline.audio(path, patient_id).findings
