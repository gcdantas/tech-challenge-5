"""Azure Speech contínuo para WAV PCM16 mono/16 kHz, com SDK injetável."""
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import threading
import wave

from .cache import ResponseCache, digest
from .config import SpeechConfig


def wav_info(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Áudio não encontrado: {path}")
    try:
        with wave.open(str(path), "rb") as f:
            if (f.getnchannels(), f.getframerate(), f.getsampwidth(), f.getcomptype()) != (1, 16000, 2, "NONE"):
                raise ValueError("Use WAV PCM 16 bits, mono, 16000 Hz")
            frames = f.getnframes()
            if frames == 0 or len(f.readframes(frames)) != frames * 2:
                raise ValueError("WAV vazio ou truncado")
            return {"sample_rate": 16000, "duration": frames / 16000}
    except (wave.Error, EOFError) as exc:
        raise ValueError("Arquivo não é WAV PCM válido") from exc


def validate_segments(segments: list, duration: float | None = None) -> list[dict]:
    result = []
    previous_end = 0.0
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
            raise ValueError("Segmento de fala com intervalo inválido")
        if start < previous_end - 0.01 or (duration is not None and end > duration + 0.1):
            raise ValueError("Segmentos sobrepostos ou fora da duração do WAV")
        if not isinstance(segment["text"], str) or not segment["text"].strip():
            raise ValueError("Segmento sem texto")
        words = segment.get("words", [])
        last_word_end = start
        for word in words:
            a, b = float(word["start"]), float(word["end"])
            if not (math.isfinite(a) and math.isfinite(b) and start - 0.01 <= a < b <= end + 0.01):
                raise ValueError("Timestamp de palavra inválido")
            if a < last_word_end - 0.01 or not word["text"].strip():
                raise ValueError("Palavras sobrepostas ou vazias")
            last_word_end = b
        result.append({**segment, "start": start, "end": end, "words": words})
        previous_end = end
    return result


class AzureSpeech:
    mock = False
    name = "azure-speech"

    def __init__(self, config: SpeechConfig, cache: ResponseCache, *, allow_azure=False, sdk=None):
        self.config, self.cache = config, cache
        self.allow_azure, self.sdk = allow_azure, sdk
        self.version = version("azure-cognitiveservices-speech") if sdk is None else "sdk-simulado"

    def transcribe(self, path: str | Path) -> dict:
        path = Path(path)
        info = wav_info(path)
        parameters = {"file_sha256": digest(path.read_bytes()), "config": self.config.model_dump(),
                      "sdk": self.version, "region": os.getenv("AZURE_SPEECH_REGION", ""),
                      "word_timestamps": True}
        result = self.cache.get_or_call("azure-speech", parameters, lambda: self._request(path))
        return {**result, "segments": validate_segments(result["segments"], info["duration"])}

    def _request(self, path: Path) -> dict:
        if not self.allow_azure:
            raise PermissionError("Cache Speech ausente. Autorize a chamada com --allow-azure ou use --mock.")
        if not os.getenv("AZURE_SPEECH_KEY") or not os.getenv("AZURE_SPEECH_REGION"):
            raise ValueError("Defina AZURE_SPEECH_KEY e AZURE_SPEECH_REGION no .env local")
        sdk = self.sdk
        if sdk is None:
            import azure.cognitiveservices.speech as sdk
        config = sdk.SpeechConfig(subscription=os.environ["AZURE_SPEECH_KEY"],
                                  region=os.environ["AZURE_SPEECH_REGION"])
        config.speech_recognition_language = self.config.language
        config.output_format = sdk.OutputFormat.Detailed
        config.request_word_level_timestamps()
        recognizer = sdk.SpeechRecognizer(speech_config=config,
                                         audio_config=sdk.audio.AudioConfig(filename=str(path)))
        done = threading.Event()
        segments, errors = [], []

        def recognized(event):
            try:
                result = event.result
                if result.reason != sdk.ResultReason.RecognizedSpeech or not result.text.strip():
                    return
                detail = json.loads(result.json)
                best = (detail.get("NBest") or [{}])[0]
                # Offset e Duration do JSON detalhado estão em ticks de 100 ns.
                start = result.offset / 10_000_000
                duration = result.duration / 10_000_000
                words = [{"text": w["Word"], "start": w["Offset"] / 10_000_000,
                          "end": (w["Offset"] + w["Duration"]) / 10_000_000}
                         for w in best.get("Words", [])]
                segments.append({"start": start, "end": start + duration,
                                 "text": result.text, "words": words,
                                 "confidence": best.get("Confidence")})
            except Exception:
                errors.append("Resposta Speech inválida")
                done.set()

        def canceled(event):
            if event.reason == sdk.CancellationReason.Error:
                # Não propagar error_details: pode conter endereço/dados da requisição.
                errors.append("Azure Speech cancelou a transcrição; verifique recurso, cota e conexão")
            done.set()

        recognizer.recognized.connect(recognized)
        recognizer.canceled.connect(canceled)
        recognizer.session_stopped.connect(lambda event: done.set())
        try:
            recognizer.start_continuous_recognition_async().get()
            if not done.wait(self.config.timeout_seconds):
                raise TimeoutError("Tempo limite da transcrição Speech excedido")
        finally:
            recognizer.stop_continuous_recognition_async().get()
        if errors:
            raise RuntimeError(errors[0])
        segments.sort(key=lambda s: s["start"])
        validate_segments(segments, wav_info(path)["duration"])
        return {"segments": segments, "provider": self.name, "sdk_version": self.version,
                "model_version": "gerenciado-pelo-serviço", "mock": False}


class MockSpeech:
    """Só aceita fixture JSON explicitamente marcada; nunca transcreve um WAV."""
    mock = True
    name = "mock-speech"
    version = "1"

    def transcribe(self, path: str | Path) -> dict:
        fixture = json.loads(Path(path).read_text(encoding="utf-8"))
        if fixture.get("mock") is not True:
            raise ValueError("Fixture Speech deve declarar mock=true")
        return {"segments": validate_segments(fixture["segments"]), "mock": True,
                "provider": self.name, "sdk_version": self.version,
                "model_version": "fixture-manual"}
