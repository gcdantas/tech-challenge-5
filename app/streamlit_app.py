"""Esqueleto Streamlit do M1 (versão provisória feita pelo I1; dono: I4).

Carrega data/patients.yaml, lê todos os findings.jsonl em outputs/, mostra uma
linha do tempo por paciente e roda a fusão mínima, gravando outputs/alerts.jsonl.
O alerta crítico também vai por e-mail (app/notify.py); sem SMTP no .env, vira .eml.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `streamlit run` coloca só app/ no sys.path
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.fusion import fuse, load_fusion_config, write_alerts  # noqa: E402
from app.loaders import (  # noqa: E402
    findings_dataframe,
    load_findings,
    load_movement,
    load_transcript,
    outputs_dir,
    patients_path,
)
from app.notify import SmtpSettings, load_alerts_config, notify, read_deliveries  # noqa: E402
from contracts.patients import load_patients  # noqa: E402

load_dotenv(ROOT / ".env", override=False)

LEVEL_LABEL = {"critico": "Crítico", "urgente": "Urgente", "atencao": "Atenção"}

st.set_page_config(page_title="Monitoramento multimodal", layout="wide")
st.title("Monitoramento multimodal de pacientes")
st.caption(
    "Protótipo de demonstração técnica. O sistema sinaliza risco e apoia a equipe; a decisão é clínica."
)

out_dir = outputs_dir()
pf = load_patients(patients_path())

# --- Fusão de todos os pacientes (grava alerts.jsonl) -------------------------
all_loaded = load_findings(out_dir)
alerts = fuse(all_loaded.findings, load_fusion_config())
# E-mail do alerta crítico: um envio por alerta, mesmo com a tela recarregando.
alerts_cfg = load_alerts_config()
alerts = notify(alerts, all_loaded.findings, out_dir, cfg=alerts_cfg, settings=SmtpSettings.from_env())
deliveries = read_deliveries(out_dir)
alerts_path = write_alerts(out_dir / "alerts.jsonl", alerts)

# --- Barra lateral ------------------------------------------------------------
with st.sidebar:
    patient_id = st.selectbox("Paciente", sorted(pf.patients))
    patient = pf.patient(patient_id)
    st.write(patient.description)
    st.markdown(f"**Referência da demo:** `{pf.demo_reference.isoformat()}`")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "artefato": name,
                    "tipo": a.kind,
                    "demo_offset (s)": a.demo_offset,
                    "início na demo": pf.demo_time(a.demo_offset).strftime("%d/%m %H:%M:%S"),
                    "arquivo": a.path or "a definir",
                }
                for name, a in patient.artifacts.items()
            ]
        ),
        hide_index=True,
    )

loaded = load_findings(out_dir, patient_id)
for err in loaded.errors:
    st.error(f"Arquivo de Findings inválido: {err}")

df = findings_dataframe(loaded.findings)
if not df.empty and df["mock"].any():
    st.warning(
        "Há Findings FICTÍCIOS (mocks do M1) na tela. Gere-os de novo com "
        "`python -m scripts.make_mocks`."
    )

# --- Alertas ------------------------------------------------------------------
st.subheader("Alertas")
patient_alerts = [a for a in alerts if a.patient_id == patient_id]
if not patient_alerts:
    st.info("Nenhum alerta para este paciente.")
_LEVEL_RANK = {"critico": 0, "urgente": 1, "atencao": 2}
for a in sorted(patient_alerts, key=lambda a: (_LEVEL_RANK[a.level], -(a.score or 0.0), a.created_at)):
    msg = (
        f"**{LEVEL_LABEL[a.level]}** · {a.created_at.strftime('%d/%m %H:%M:%S')} · `{a.rule}`"
        + (f" · score {a.score:.2f}" if a.score is not None else "")
        + f"  \n{a.description}  \nFindings: {len(a.finding_ids)}"
    )
    if a.suppressed_finding_ids:
        msg += f" (+{len(a.suppressed_finding_ids)} do mesmo tipo dentro do cooldown, sem novo alerta)"
    d = deliveries.get(a.alert_id)
    if d is not None and d.sent_at is not None:
        msg += (
            f"  \nE-mail enviado à equipe às {d.sent_at.strftime('%H:%M:%S')} "
            f"({len(d.recipients)} destinatário(s))."
        )
    elif d is not None:
        msg += f"  \nE-mail não enviado ({d.error}). Cópia de contingência: `{d.eml_path}`."
    {"critico": st.error, "urgente": st.warning, "atencao": st.info}[a.level](msg)
st.caption(
    f"Alertas gravados em `{alerts_path}`. Níveis que geram e-mail: "
    f"{', '.join(alerts_cfg.email.levels) if alerts_cfg.email.enabled else 'nenhum (desligado)'}."
)

# --- Linha do tempo -----------------------------------------------------------
st.subheader("Linha do tempo multimodal")
if df.empty:
    st.info("Nenhum Finding encontrado em outputs/. Rode `python -m scripts.make_mocks`.")
else:
    tl = df.copy()
    # Eventos pontuais ganham 30 s de largura só para aparecer no gráfico.
    tl["fim_plot"] = tl[["inicio", "fim"]].max(axis=1)
    tl.loc[tl["fim_plot"] == tl["inicio"], "fim_plot"] = tl["inicio"] + pd.Timedelta(seconds=30)
    # Altair/Vega trabalham em UTC; o eixo mostra o horário local do navegador.
    tl["inicio"] = pd.to_datetime(tl["inicio"], utc=True)
    tl["fim_plot"] = pd.to_datetime(tl["fim_plot"], utc=True)
    chart = (
        alt.Chart(tl)
        .mark_bar(size=14, cornerRadius=3)
        .encode(
            x=alt.X("inicio:T", title="Horário"),
            x2="fim_plot:T",
            y=alt.Y("origem:N", title="Origem", sort=["video", "movement", "prescription", "audio", "text", "vitals"]),
            color=alt.Color(
                "severidade:N",
                scale=alt.Scale(
                    domain=["low", "medium", "high", "critical"],
                    range=["#7fb3d5", "#f5b041", "#e67e22", "#c0392b"],
                ),
            ),
            tooltip=["origem", "tipo", "severidade", "score", "descricao"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(
        df.drop(columns=["finding_id"]).assign(
            inicio=df["inicio"].map(lambda t: t.isoformat()), fim=df["fim"].map(lambda t: t.isoformat())
        ),
        hide_index=True,
    )

# --- Detalhes por modalidade -------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Série de movimentação (Integrante 1 → Integrante 3)")
    mov = load_movement(out_dir, patient_id)
    if mov is None:
        st.info("movement.parquet não encontrado.")
    else:
        mov = mov.assign(timestamp=pd.to_datetime(mov["timestamp"], utc=True))
        st.line_chart(mov.set_index("timestamp")[["centroid_x", "centroid_y"]])
        st.area_chart(mov.set_index("timestamp")["in_bed"].astype(int), height=120)
with col2:
    st.subheader("Transcrição da consulta (Integrante 2)")
    tr = load_transcript(out_dir, patient_id)
    if tr is None:
        st.info("transcript.json não encontrado.")
    else:
        if tr.get("mock"):
            st.caption("Transcrição fictícia (mock).")
        for seg in tr.get("segments", []):
            text = seg["text"]
            for term in seg.get("critical_terms", []):
                text = text.replace(term["term"], f"**:red[{term['term']}]**")
            st.markdown(f"`{seg['timestamp_start'][11:19]}` {text}")
