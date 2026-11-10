"""Gera os mocks da semana 1 (marco M1) de I1, I2 e I3 em outputs/.

TODOS OS DADOS GERADOS AQUI SÃO FICTÍCIOS. Servem só para o pipeline rodar
de ponta a ponta antes dos módulos reais. Marcação:
- Findings: ``description`` começa com "[MOCK]", ``model.name`` começa com
  "mock-" e ``evidence.mock`` = true;
- movement.parquet: metadados do arquivo ``mock=true`` (as colunas seguem o plano);
- transcript.json: campo ``mock`` = true.

Os horários seguem o cenário P001 do plano v5, a partir do demo_reference e
dos demo_offset de data/patients.yaml. Os Findings da regra crítica
(indicador vocal + termo respiratório + queda de SpO2) caem na mesma janela
de 30 min.

Uso:
    python -m scripts.make_mocks [--out outputs] [--patients data/patients.yaml] [--skip i1 ...]

Use ``--skip i1`` depois de rodar o módulo de vídeo real, para não sobrescrever
outputs/video/P001/ (findings.jsonl e movement.parquet reais).
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from contracts.finding import Finding, ModelInfo, write_findings
from contracts.patients import DEFAULT_PATIENTS_PATH, REPO_ROOT, PatientsFile, load_patients

PATIENT = "P001"
MOCK_PREFIX = "[MOCK] "
MOVEMENT_COLUMNS = [
    "patient_id", "timestamp", "centroid_x", "centroid_y", "torso_angle", "velocidade", "in_bed",
]
MOVEMENT_HZ = 5.0


class _Ids:
    """UUID4 determinísticos, para que os mocks sejam idênticos a cada execução."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def __call__(self) -> uuid.UUID:
        return uuid.UUID(int=self._rng.getrandbits(128), version=4)


def _mock_finding(
    ids: _Ids,
    *,
    source: str,
    type_: str,
    start: datetime,
    end: datetime,
    score: float,
    severity: str,
    description: str,
    evidence: dict[str, Any],
    owner: str,
) -> Finding:
    return Finding(
        finding_id=ids(),
        patient_id=PATIENT,
        source=source,
        type=type_,
        timestamp_start=start,
        timestamp_end=end,
        score=score,
        severity=severity,
        description=MOCK_PREFIX + description,
        evidence={**evidence, "mock": True},
        model=ModelInfo(name=f"mock-{owner}", version="0"),
        artifacts=[],
    )


# --------------------------------------------------------------------------- I1


def movement_mock(pf: PatientsFile, duration_s: float = 60.0, exit_at_s: float = 40.0) -> pd.DataFrame:
    """Série sintética a 5 Hz: paciente parado no leito e, a partir de ``exit_at_s``, saindo.

    Unidades (provisórias): centroide normalizado 0–1 no quadro; torso_angle em graus
    em relação à vertical (0 = em pé, 90 = deitado); velocidade em unidades
    normalizadas por segundo.
    """
    rng = np.random.default_rng(7)
    t = np.arange(0.0, duration_s, 1.0 / MOVEMENT_HZ)
    n = len(t)

    # Leito: deitado, quase parado.
    cx = 0.42 + rng.normal(0, 0.002, n)
    cy = 0.58 + rng.normal(0, 0.002, n)
    torso = 86.0 + rng.normal(0, 1.0, n)

    # Senta (exit_at → +3 s), levanta (+3 → +6 s) e caminha para fora do leito.
    sit_end, stand_end = exit_at_s + 3.0, exit_at_s + 6.0
    sitting = (t >= exit_at_s) & (t < sit_end)
    frac = (t[sitting] - exit_at_s) / 3.0
    torso[sitting] = 86.0 - 66.0 * frac
    cx[sitting] = 0.42 + 0.18 * frac

    standing = (t >= sit_end) & (t < stand_end)
    frac = (t[standing] - sit_end) / 3.0
    torso[standing] = 20.0 - 12.0 * frac
    cx[standing] = 0.60 + 0.12 * frac
    cy[standing] = 0.58 - 0.18 * frac

    walking = t >= stand_end
    frac = (t[walking] - stand_end) / max(duration_s - stand_end, 1e-9)
    torso[walking] = 8.0 + rng.normal(0, 2.0, int(walking.sum()))
    cx[walking] = 0.72 + 0.23 * frac
    cy[walking] = 0.40 + rng.normal(0, 0.003, int(walking.sum()))

    # Fora do leito quando o centroide passa da borda do leito (x > 0.62 no mock).
    in_bed = cx <= 0.62

    vel = np.zeros(n)
    vel[1:] = np.hypot(np.diff(cx), np.diff(cy)) * MOVEMENT_HZ

    offset = pf.artifact(PATIENT, "video_leito").demo_offset
    timestamps = [pf.demo_time(offset, float(s)) for s in t]
    return pd.DataFrame(
        {
            "patient_id": PATIENT,
            "timestamp": pd.DatetimeIndex(timestamps),
            "centroid_x": cx.round(4),
            "centroid_y": cy.round(4),
            "torso_angle": torso.round(2),
            "velocidade": vel.round(4),
            "in_bed": in_bed,
        },
        columns=MOVEMENT_COLUMNS,
    )


def write_movement_parquet(df: pd.DataFrame, path: Path, extra_meta: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({k.encode(): v.encode() for k, v in extra_meta.items()})
    pq.write_table(table.replace_schema_metadata(meta), path)
    return path


def video_findings_mock(pf: PatientsFile, ids: _Ids) -> list[Finding]:
    def t(s: float) -> datetime:
        return pf.artifact_time(PATIENT, "video_fisio", s)

    return [
        _mock_finding(
            ids, source="video", type_="amplitude_incompleta", start=t(12.0), end=t(18.4),
            score=0.62, severity="medium",
            description="Amplitude incompleta na repetição 3: flexão de joelho até 128°, esperado até 100°",
            evidence={"metric": "knee_angle_min", "value": 128, "expected": [70, 100], "rep": 3},
            owner="i1",
        ),
        _mock_finding(
            ids, source="video", type_="postura_fora_padrao", start=t(31.0), end=t(36.2),
            score=0.55, severity="low",
            description="Inclinação do tronco de 52° na repetição 6, acima da faixa esperada",
            evidence={"metric": "trunk_angle_max", "value": 52, "expected": [0, 45], "rep": 6},
            owner="i1",
        ),
    ]


# --------------------------------------------------------------------------- I2

TRANSCRIPT_SEGMENTS = [
    # (início s, fim s, texto, sentimento, termos críticos)
    (0.0, 5.0, "Bom dia. Como o senhor passou a noite?", "neutral", []),
    (5.5, 12.0, "Mais ou menos, doutor. Acordei várias vezes.", "negative", []),
    (12.5, 18.0, "Teve dor em algum lugar ou febre?", "neutral", []),
    (18.5, 26.0, "Febre não. Só um cansaço que não passa.", "negative", []),
    (60.0, 66.0, "E para caminhar até o banheiro, como foi?", "neutral", []),
    (75.0, 79.0, "Ontem à noite senti muita falta de ar quando fui ao banheiro.", "negative",
     ["falta de ar"]),
    (80.0, 86.0, "Certo. Vou pedir para a enfermagem acompanhar a saturação.", "neutral", []),
]


def transcript_mock(pf: PatientsFile) -> dict[str, Any]:
    art = pf.artifact(PATIENT, "consulta")
    segments = []
    for start, end, text, sentiment, terms in TRANSCRIPT_SEGMENTS:
        segments.append(
            {
                "start": start,
                "end": end,
                "timestamp_start": pf.demo_time(art.demo_offset, start).isoformat(),
                "timestamp_end": pf.demo_time(art.demo_offset, end).isoformat(),
                "text": text,
                "sentiment": sentiment,
                "critical_terms": [
                    {"term": term, "offset": text.index(term), "length": len(term)} for term in terms
                ],
            }
        )
    return {
        "mock": True,
        "aviso": "Transcrição FICTÍCIA escrita à mão para o mock da semana 1 (I2).",
        "patient_id": PATIENT,
        "audio": art.path,
        "language": "pt-BR",
        "segments": segments,
    }


def audio_findings_mock(pf: PatientsFile, ids: _Ids) -> list[Finding]:
    def t(s: float) -> datetime:
        return pf.artifact_time(PATIENT, "consulta", s)

    return [
        _mock_finding(
            ids, source="audio", type_="termo_critico", start=t(75.0), end=t(79.0),
            score=0.8, severity="medium",
            description='Termo crítico "falta de ar" na fala do paciente',
            evidence={"term": "falta de ar", "segment_text": TRANSCRIPT_SEGMENTS[5][2]},
            owner="i2",
        ),
        _mock_finding(
            ids, source="audio", type_="indicador_vocal_alterado", start=t(30.0), end=t(90.0),
            score=0.68, severity="medium",
            description=(
                "Indicador experimental de alteração vocal, baseado em características "
                "acústicas, acima do limiar em relação à linha de base do próprio paciente"
            ),
            evidence={
                "baseline": "primeiros 30 s da consulta",
                "zscores": {"taxa_fala": -2.3, "pausas": 2.6, "pitch": -1.1, "jitter": 1.9, "shimmer": 2.1},
            },
            owner="i2",
        ),
    ]


def text_findings_mock(pf: PatientsFile, ids: _Ids) -> list[Finding]:
    t = pf.artifact_time(PATIENT, "nota_recente")
    return [
        _mock_finding(
            ids, source="text", type_="termo_critico", start=t, end=t,
            score=0.75, severity="medium",
            description='Termo crítico "dispneia aos esforços" na nota de evolução mais recente',
            evidence={"term": "dispneia aos esforços", "note": "nota_recente", "ausente_na_anterior": True},
            owner="i2",
        ),
    ]


# --------------------------------------------------------------------------- I3


def anomaly_findings_mock(pf: PatientsFile, ids: _Ids, movement: pd.DataFrame) -> list[Finding]:
    vit = pf.artifact(PATIENT, "vitais").demo_offset
    presc = pf.artifact(PATIENT, "prescricoes").demo_offset

    # Saída do leito coerente com o movement.parquet mock: primeira amostra fora do leito.
    out = movement.loc[~movement["in_bed"], "timestamp"]
    exit_ts = out.iloc[0].to_pydatetime()

    return [
        _mock_finding(
            ids, source="vitals", type_="dessaturacao",
            start=pf.demo_time(vit, 45 * 60), end=pf.demo_time(vit, 48 * 60),
            score=0.9, severity="high",
            description="Queda de SpO2 para 87% sustentada por 3 min",
            evidence={"metric": "spo2", "value": 87, "expected": [94, 100]},
            owner="i3",
        ),
        _mock_finding(
            ids, source="prescription", type_="dose_duplicada",
            start=pf.demo_time(presc, 20 * 60), end=pf.demo_time(presc, 20 * 60),
            score=0.7, severity="medium",
            description="Mesma medicação e dose prescritas duas vezes para o mesmo horário",
            evidence={"drug": "dipirona", "dose_mg": 1000, "count": 2},
            owner="i3",
        ),
        _mock_finding(
            ids, source="movement", type_="saida_do_leito",
            start=exit_ts, end=exit_ts + timedelta(seconds=6),
            score=0.7, severity="medium",
            description="Saída do leito na série de movimentação",
            evidence={"in_bed_before": True, "in_bed_after": False},
            owner="i3",
        ),
    ]


# --------------------------------------------------------------------------- main


OWNERS = ("i1", "i2", "i3")


def make_all(
    out_dir: str | Path = "outputs",
    patients_path: str | Path = DEFAULT_PATIENTS_PATH,
    skip: tuple[str, ...] | list[str] = (),
) -> dict[str, Path]:
    """Gera os mocks (exceto os dos integrantes em ``skip``) e devolve {nome: caminho}."""
    out = Path(out_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    pf = load_patients(patients_path)
    ids = _Ids(seed=2026)

    movement = movement_mock(pf)
    written: dict[str, Path] = {}

    # Os IDs são gerados sempre na mesma ordem, para que pular um integrante não
    # mude os IDs dos outros.
    video = video_findings_mock(pf, ids)
    audio = audio_findings_mock(pf, ids)
    text = text_findings_mock(pf, ids)
    anomaly = anomaly_findings_mock(pf, ids, movement)

    if "i1" not in skip:
        written["video_findings"] = write_findings(out / "video" / PATIENT / "findings.jsonl", video)
        written["movement"] = write_movement_parquet(
            movement,
            out / "video" / PATIENT / "movement.parquet",
            {"mock": "true", "generator": "scripts/make_mocks.py", "aviso": "Série FICTÍCIA (mock da semana 1, I1)"},
        )
    if "i2" not in skip:
        written["audio_findings"] = write_findings(out / "audio" / PATIENT / "findings.jsonl", audio)
        transcript_path = out / "audio" / PATIENT / "transcript.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(json.dumps(transcript_mock(pf), ensure_ascii=False, indent=2), encoding="utf-8")
        written["transcript"] = transcript_path
        written["text_findings"] = write_findings(out / "text" / PATIENT / "findings.jsonl", text)
    if "i3" not in skip:
        written["anomaly_findings"] = write_findings(out / "anomaly" / PATIENT / "findings.jsonl", anomaly)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gera os mocks FICTÍCIOS do M1 em outputs/.")
    parser.add_argument("--out", default="outputs", help="pasta de saída (padrão: outputs)")
    parser.add_argument("--patients", default=str(DEFAULT_PATIENTS_PATH), help="caminho do patients.yaml")
    parser.add_argument("--skip", nargs="*", choices=OWNERS, default=[],
                        help="não gera os mocks destes integrantes (ex.: --skip i1)")
    args = parser.parse_args(argv)
    for name, path in make_all(args.out, args.patients, args.skip).items():
        print(f"{name:18s} {path}")


if __name__ == "__main__":
    main()
