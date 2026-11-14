"""Fusão: combinação ponderada dos scores e deduplicação/cooldown por paciente e tipo.

Findings SINTÉTICOS montados aqui; o cenário P001 com mocks fica em test_p001_smoke.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from app.fusion import WEIGHTED_RULE, FusionConfig, fuse, load_fusion_config, weighted_score
from contracts.finding import DEMO_TZ, Finding, ModelInfo

T0 = datetime(2026, 11, 3, 10, 0, tzinfo=DEMO_TZ)


def mk(source, type_, minute, score=0.5, severity="medium", patient="P001", **evidence) -> Finding:
    start = T0 + timedelta(minutes=minute)
    return Finding(
        finding_id=uuid.uuid4(),
        patient_id=patient,
        source=source,
        type=type_,
        timestamp_start=start,
        timestamp_end=start + timedelta(seconds=30),
        score=score,
        severity=severity,
        description=f"teste {source}:{type_}",
        evidence=evidence,
        model=ModelInfo(name="teste", version="0"),
    )


@pytest.fixture
def cfg() -> FusionConfig:
    return load_fusion_config()


def by_rule(alerts):
    out = {}
    for a in alerts:
        out.setdefault(a.rule, []).append(a)
    return out


# --------------------------------------------------------------- combinação ponderada


def test_config_do_repositorio_tem_combinacao_ponderada(cfg):
    w = cfg.weighted
    assert w.enabled and w.min_sources >= 2
    assert "critico" not in w.thresholds  # crítico só por regra ou severidade
    assert set(w.source_weights) == {"video", "audio", "text", "vitals", "prescription", "movement"}


def test_score_ponderado_usa_o_maior_score_de_cada_fonte(cfg):
    fs = [mk("vitals", "dessaturacao", 0, 0.9), mk("vitals", "dessaturacao", 5, 0.4), mk("video", "x", 1, 0.5)]
    wv, wvid = cfg.weighted.source_weights["vitals"], cfg.weighted.source_weights["video"]
    assert weighted_score(fs, cfg) == pytest.approx(round((wv * 0.9 + wvid * 0.5) / (wv + wvid), 3))
    assert weighted_score(fs[:1], cfg) == 0.9  # uma fonte: o próprio score


def test_duas_fontes_na_janela_viram_um_alerta_ponderado(cfg):
    fs = [mk("movement", "saida_do_leito", 0, 0.5), mk("prescription", "dose_duplicada", 10, 0.5)]
    (alert,) = fuse(fs, cfg)
    assert alert.rule == WEIGHTED_RULE
    assert alert.level == "atencao"
    assert sorted(alert.finding_ids) == sorted(str(f.finding_id) for f in fs)
    assert alert.score == pytest.approx(0.5)
    assert "não é probabilidade clínica" in alert.description


def test_corroboracao_de_scores_altos_sobe_para_urgente(cfg):
    """Dois Findings 'medium' (atenção sozinhos) de fontes diferentes com score alto."""
    fs = [mk("vitals", "dessaturacao", 0, 0.85), mk("text", "termo_critico", 12, 0.8, term="tosse")]
    (alert,) = fuse(fs, cfg)
    assert alert.rule == WEIGHTED_RULE and alert.level == "urgente"


def test_combinacao_nunca_rebaixa_severidade(cfg):
    fs = [mk("vitals", "dessaturacao", 0, 0.4, "high"), mk("video", "postura_fora_padrao", 3, 0.3, "low")]
    (alert,) = fuse(fs, cfg)
    assert alert.rule == WEIGHTED_RULE and alert.level == "urgente"


def test_combinacao_nao_gera_critico_so_pelo_score(cfg):
    fs = [mk("vitals", "dessaturacao", 0, 1.0, "high"), mk("prescription", "dose_duplicada", 1, 1.0, "high")]
    (alert,) = fuse(fs, cfg)
    assert alert.score == 1.0 and alert.level == "urgente"


def test_fora_da_janela_nao_combina(cfg):
    fs = [mk("movement", "saida_do_leito", 0), mk("prescription", "dose_duplicada", 31)]
    rules = sorted(a.rule for a in fuse(fs, cfg))
    assert rules == ["isolado:movement:saida_do_leito", "isolado:prescription:dose_duplicada"]


def test_uma_fonte_so_nao_combina(cfg):
    fs = [mk("video", "amplitude_incompleta", 0), mk("video", "postura_fora_padrao", 1)]
    assert all(a.rule.startswith("isolado:video:") for a in fuse(fs, cfg))


def test_score_abaixo_do_limiar_fica_isolado(cfg):
    fs = [mk("movement", "saida_do_leito", 0, 0.1, "low"), mk("video", "postura_fora_padrao", 1, 0.1, "low")]
    assert all(a.rule.startswith("isolado:") for a in fuse(fs, cfg))


def test_combinacao_desligada(cfg):
    cfg = cfg.model_copy(update={"weighted": cfg.weighted.model_copy(update={"enabled": False})})
    fs = [mk("movement", "saida_do_leito", 0), mk("prescription", "dose_duplicada", 10)]
    assert all(a.rule.startswith("isolado:") for a in fuse(fs, cfg))


def test_regra_explicita_tem_precedencia_sobre_a_ponderada(cfg):
    fs = [
        mk("audio", "indicador_vocal_alterado", 0, 0.7),
        mk("audio", "termo_critico", 1, 0.8, term="falta de ar"),
        mk("vitals", "dessaturacao", 15, 0.9, "high"),
        mk("movement", "saida_do_leito", 16, 0.6),
    ]
    alerts = by_rule(fuse(fs, cfg))
    (crit,) = alerts["respiratorio_vocal_spo2"]
    assert crit.level == "critico" and len(crit.finding_ids) == 3
    assert crit.score is not None
    # A saída do leito sobra sozinha (os outros já foram usados pela regra).
    assert list(alerts) == ["respiratorio_vocal_spo2", "isolado:movement:saida_do_leito"]


@pytest.mark.parametrize(
    "bad",
    [
        {"source_weights": {"vitals": 0}},
        {"source_weights": {"desconhecida": 1}},
        {"thresholds": {"urgente": 1.5}},
        {"min_sources": 1},
    ],
)
def test_config_ponderada_invalida(cfg, bad):
    raw = cfg.model_dump()
    raw["weighted"].update(bad)
    with pytest.raises(ValidationError):
        FusionConfig.model_validate(raw)


# --------------------------------------------------------------- deduplicação e cooldown


def test_cooldown_por_paciente_e_tipo_deduplica_isolados(cfg):
    fs = [mk("video", "amplitude_incompleta", m, 0.4) for m in (0, 5, 10, 29)]
    (alert,) = fuse(fs, cfg)
    assert alert.finding_ids == [str(fs[0].finding_id)]
    assert alert.suppressed_finding_ids == [str(f.finding_id) for f in fs[1:]]


def test_depois_do_cooldown_alerta_de_novo(cfg):
    fs = [mk("video", "amplitude_incompleta", 0), mk("video", "amplitude_incompleta", 31)]
    assert len(fuse(fs, cfg)) == 2


def test_tipos_diferentes_nao_se_deduplicam(cfg):
    fs = [mk("video", "amplitude_incompleta", 0), mk("video", "postura_fora_padrao", 1)]
    assert len(fuse(fs, cfg)) == 2


def test_pacientes_diferentes_nao_se_deduplicam(cfg):
    fs = [mk("video", "amplitude_incompleta", 0, patient="P001"), mk("video", "amplitude_incompleta", 1, patient="P002")]
    assert sorted(a.patient_id for a in fuse(fs, cfg)) == ["P001", "P002"]


def test_escalonamento_ignora_cooldown(cfg):
    fs = [mk("video", "amplitude_incompleta", 0, 0.3, "low"), mk("video", "amplitude_incompleta", 5, 0.8, "high")]
    alerts = fuse(fs, cfg)
    assert [a.level for a in alerts] == ["atencao", "urgente"]
    # E o que vem depois, no mesmo nível ou abaixo, é deduplicado no urgente.
    fs.append(mk("video", "amplitude_incompleta", 6, 0.3, "low"))
    alerts = fuse(fs, cfg)
    assert [a.level for a in alerts] == ["atencao", "urgente"]
    assert alerts[1].suppressed_finding_ids == [str(fs[2].finding_id)]


def test_regra_dentro_do_cooldown_absorve_os_findings(cfg):
    """Segunda queda de SpO2 dentro do cooldown: não vira alerta isolado nem crítico novo."""
    fs = [
        mk("audio", "indicador_vocal_alterado", 0, 0.7),
        mk("audio", "termo_critico", 1, 0.8, term="falta de ar"),
        mk("vitals", "dessaturacao", 10, 0.9, "high"),
        mk("vitals", "dessaturacao", 20, 0.9, "high"),
    ]
    (crit,) = fuse(fs, cfg)
    assert crit.level == "critico"
    assert crit.suppressed_finding_ids == [str(fs[3].finding_id)]


def test_alert_id_estavel_quando_chegam_findings_deduplicados(cfg):
    """O e-mail usa o alert_id para não reenviar: ele não pode mudar com os suprimidos."""
    fs = [mk("video", "amplitude_incompleta", 0)]
    first = fuse(fs, cfg)[0].alert_id
    fs.append(mk("video", "amplitude_incompleta", 3))
    assert fuse(fs, cfg)[0].alert_id == first


def test_fusao_independe_da_ordem_de_entrada(cfg):
    fs = [
        mk("video", "amplitude_incompleta", 0),
        mk("movement", "saida_do_leito", 4),
        mk("video", "amplitude_incompleta", 6),
        mk("prescription", "dose_duplicada", 50),
    ]
    a = [x.model_dump() for x in fuse(fs, cfg)]
    b = [x.model_dump() for x in fuse(list(reversed(fs)), cfg)]
    assert a == b
