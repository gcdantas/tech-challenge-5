"""Material do vídeo para o relatório e para a edição do vídeo de entrega.

    python -m scripts.material_video            # figuras + pacote
    python -m scripts.material_video --figuras  # só as imagens do README
    python -m scripts.material_video --pacote   # só o pacote para o Integrante 4

Pré-requisito: o módulo de vídeo já rodou para o P001 (README, seção 7.3):
fisioterapia (REHAB24-6 PM_105), leito (Charades SSP8C), OpenPose e, para a
figura de comparação, notebooks.comparacao.openpose_yolo.

- figuras -> docs/img/video/ (versionadas; quadros parados citados no README,
  como as licenças dos datasets permitem em publicação acadêmica).
- pacote  -> outputs/pacote_video/ (fora do Git; a licença do Charades proíbe
  redistribuir os dados, então cada integrante gera a própria cópia): clipes dos
  desvios usados na demo, vídeos anotados, quadros e um LEIAME.md com o índice.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P001 = ROOT / "outputs" / "video" / "P001"
COMPARACAO = ROOT / "outputs" / "comparacao"
FIGURAS = ROOT / "docs" / "img" / "video"
PACOTE = ROOT / "outputs" / "pacote_video"
PHYSIO_STEM = "PM_105-Camera18-30fps-transposed"
BED_STEM = "SSP8C"
EXIT_S = 18.0  # saída estável do leito no SSP8C (README, seção 7.8)
CITACAO = (
    "Fisioterapia: REHAB24-6 (Černek, Sedmidubsky e Budikova, SISAP 2024; CC BY-NC 4.0). "
    "Leito: Charades (Sigurdsson et al., ECCV 2016; licença não comercial do Allen Institute for AI)."
)


def _run(scenario: str, stem: str) -> dict:
    path = P001 / "runs" / f"{scenario}_{stem}.json"
    if not path.exists():
        raise SystemExit(f"erro: falta {path}. Rode o módulo de vídeo para o P001 (README, seção 7.3).")
    return json.loads(path.read_text(encoding="utf-8"))


def _abs(rel: str) -> Path:
    return ROOT / rel


def _event(run: dict, type_: str, rep: int) -> dict:
    for e in run["events"]:
        if e["type"] == type_ and e["rep"] == rep:
            return e
    raise SystemExit(f"erro: não há desvio {type_} na repetição {rep} do {run['video_stem']}")


def _resize(img: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def _save(path: Path, img: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise SystemExit(f"erro ao gravar {path}")
    path.write_bytes(buf.tobytes())
    return path


def _side_by_side(a: np.ndarray, b: np.ndarray, height: int = 360) -> np.ndarray:
    def fit(x):
        return cv2.resize(x, (int(x.shape[1] * height / x.shape[0]), height), interpolation=cv2.INTER_AREA)

    gap = np.full((height, 8, 3), 255, np.uint8)
    return np.hstack([fit(a), gap, fit(b)])


def _deepest_openpose_frame(physio: dict, openpose: dict) -> dict:
    """Quadro do OpenPose mais próximo do maior agachamento (menor ângulo do joelho)."""
    t = np.array(physio["series"]["t"], float)
    knee = np.array([np.nan if v is None else v for v in physio["series"]["primary"]], float)

    def knee_at(fr):
        i = int(np.argmin(np.abs(t - fr["t"])))
        return knee[i] if np.isfinite(knee[i]) else 999.0

    return min(openpose["frames"], key=knee_at)


def _bed_frames(bed: dict) -> tuple[Path, Path]:
    fps = bed["video_info"]["fps"]
    frames = sorted(_abs(p) for p in bed["frames"])
    idx = [int(p.stem.split("_")[-1]) for p in frames]
    before = frames[0]
    after = next(p for p, i in zip(frames, idx) if i / fps >= EXIT_S + 1.0)
    return before, after


def angle_chart(physio: dict, out: Path, marks: dict[int, str]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = physio["series"]
    t = np.array(s["t"], float)
    knee = np.array([np.nan if v is None else v for v in s["primary"]], float)
    trunk = np.array([np.nan if v is None else v for v in s["trunk"]], float)
    amp_lim = physio["thresholds"]["amplitude_incompleta"]["expected"][1]
    post_lim = physio["thresholds"]["postura_fora_padrao"]["expected"][1]

    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=110)
    for e in physio["events"]:
        color = "#c0392b" if e["type"] == "amplitude_incompleta" else "#8e44ad"
        ax.axvspan(e["t_start"], e["t_end"], color=color, alpha=0.12, lw=0)
    ax.plot(t, knee, color="#2e86c1", lw=1.6, label="ângulo do joelho")
    ax.plot(t, trunk, color="#d68910", lw=1.6, label="inclinação do tronco")
    ax.axhline(amp_lim, color="#2e86c1", ls="--", lw=1, label=f"limite do joelho ({amp_lim:.0f}°)")
    ax.axhline(post_lim, color="#d68910", ls="--", lw=1, label=f"limite do tronco ({post_lim:.0f}°)")
    for e in physio["events"]:
        if e["rep"] in marks and e["type"] == marks[e["rep"]]:
            ax.annotate(f"rep. {e['rep']}", (e["t_ref"], 175), ha="center", fontsize=8)
    ax.set_xlabel("tempo no vídeo (s)")
    ax.set_ylabel("graus")
    ax.set_ylim(0, 185)
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=4, frameon=False)
    ax.set_title("PM_105 (REHAB24-6): faixas vermelhas = amplitude incompleta; roxas = postura fora da faixa", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def figuras(amp_rep: int, post_rep: int) -> list[Path]:
    physio = _run("physio", PHYSIO_STEM)
    openpose = _run("openpose", PHYSIO_STEM)
    bed = _run("bed", BED_STEM)
    out = []
    for name, type_, rep in (("fisio_amplitude", "amplitude_incompleta", amp_rep),
                             ("fisio_postura", "postura_fora_padrao", post_rep)):
        e = _event(physio, type_, rep)
        out.append(_save(FIGURAS / f"{name}_rep{rep}.jpg", _resize(cv2.imread(str(_abs(e["frame"]))))))
    op = _deepest_openpose_frame(physio, openpose)
    out.append(_save(FIGURAS / "openpose_PM_105.jpg", _resize(cv2.imread(str(_abs(op["image"]))))))
    comp = COMPARACAO / "PM_105_comparacao.jpg"
    if comp.exists():
        out.append(_save(FIGURAS / "comparacao_openpose_yolo_PM_105.jpg", _resize(cv2.imread(str(comp)))))
    else:
        print(f"aviso: {comp} não existe (rode notebooks.comparacao.openpose_yolo); figura pulada")
    before, after = _bed_frames(bed)
    out.append(_save(FIGURAS / "leito_SSP8C.jpg", _side_by_side(cv2.imread(str(before)), cv2.imread(str(after)))))
    out.append(angle_chart(physio, FIGURAS / "angulos_PM_105.png",
                           {amp_rep: "amplitude_incompleta", post_rep: "postura_fora_padrao"}))
    return out


def pacote(amp_rep: int, post_rep: int) -> Path:
    physio = _run("physio", PHYSIO_STEM)
    openpose = _run("openpose", PHYSIO_STEM)
    bed = _run("bed", BED_STEM)
    if PACOTE.exists():
        shutil.rmtree(PACOTE)
    (PACOTE / "clipes").mkdir(parents=True)
    (PACOTE / "videos").mkdir()
    (PACOTE / "quadros" / "openpose").mkdir(parents=True)

    def cp(src: Path, dst: Path) -> str:
        shutil.copy2(src, dst)
        return dst.relative_to(PACOTE).as_posix()

    linhas = []
    for label, type_, rep in (("amplitude incompleta", "amplitude_incompleta", amp_rep),
                              ("postura fora da faixa", "postura_fora_padrao", post_rep)):
        e = _event(physio, type_, rep)
        clip = cp(_abs(e["clip"]), PACOTE / "clipes" / f"desvio_{type_}_rep{rep}.mp4")
        frame = cp(_abs(e["frame"]), PACOTE / "quadros" / f"desvio_{type_}_rep{rep}.jpg")
        linhas.append(f"| {label}, repetição {rep} | {e['timestamp_start'][11:19]} | {e['severity']} | "
                      f"{e['description']} | `{clip}`, `{frame}` |")
    v_fisio = cp(_abs(physio["annotated_video"]), PACOTE / "videos" / "fisioterapia_PM_105_anotado.mp4")
    v_leito = cp(_abs(bed["annotated_video"]), PACOTE / "videos" / "leito_SSP8C_anotado.mp4")
    for fr in openpose["frames"]:
        cp(_abs(fr["image"]), PACOTE / "quadros" / "openpose" / Path(fr["image"]).name)
    before, after = _bed_frames(bed)
    q_antes = cp(before, PACOTE / "quadros" / "leito_antes_da_saida.jpg")
    q_depois = cp(after, PACOTE / "quadros" / "leito_depois_da_saida.jpg")
    extras = []
    for src, name in ((COMPARACAO / "PM_105_comparacao.jpg", "comparacao_openpose_yolo_PM_105.jpg"),
                      (FIGURAS / "angulos_PM_105.png", "angulos_PM_105.png")):
        if src.exists():
            extras.append(cp(src, PACOTE / "quadros" / name))

    leia = f"""# Pacote do vídeo (Integrante 1) para a edição do vídeo de entrega

Gerado por `python -m scripts.material_video --pacote`. Bloco do Integrante 1
no roteiro: **1:00–3:30**.

> Uso só dentro do grupo e na entrega acadêmica. {CITACAO}
> O Charades proíbe redistribuir os dados: não publique este pacote; no vídeo
> de entrega use só trechos curtos, com a citação na tela ou na descrição.

## Desvios para mostrar (paciente sintético P001)

| Desvio | Horário na demo | Severidade | Descrição | Arquivos |
|---|---|---|---|---|
{chr(10).join(linhas)}

Os dois desvios estão em repetições anotadas como **incorretas** no REHAB24-6.
O vídeo também tem falsos positivos de severidade baixa (repetições 6–10, em
meio-perfil): se aparecerem, dizer que existem e por quê (README, seção 7.12).

## Demais arquivos

| Arquivo | O que mostra |
|---|---|
| `{v_fisio}` | sessão inteira com esqueleto do YOLOv8-pose, ângulos, repetição e desvio ativo |
| `{v_leito}` | leito com pessoa, leito e zona da borda (área crítica); saída estável aos 18 s (10:10:18 na demo) |
| `quadros/openpose/` | {len(openpose['frames'])} quadros com o esqueleto do OpenPose (requisito do enunciado) |
| `{q_antes}`, `{q_depois}` | leito antes e depois da saída |
{chr(10).join(f"| `{x}` | {'comparação OpenPose × YOLOv8-pose com o gabarito' if 'comparacao' in x else 'ângulos ao longo do vídeo, com os desvios marcados'} |" for x in extras)}

Relatório automático completo: `outputs/video/P001/report.html` (abrir no
navegador; mostrar no vídeo como "relatório automático de desvios").

Linguagem obrigatória ao narrar: "o sistema sinaliza risco e apoia a equipe; a
decisão é clínica"; "paciente sintético P001, cujos dados vêm de fontes
distintas associadas só para a demonstração técnica".
"""
    (PACOTE / "LEIAME.md").write_text(leia, encoding="utf-8")
    return PACOTE


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Figuras do README e pacote do vídeo para o Integrante 4.")
    ap.add_argument("--figuras", action="store_true")
    ap.add_argument("--pacote", action="store_true")
    ap.add_argument("--amplitude-rep", type=int, default=12, help="repetição do desvio de amplitude (padrão: 12)")
    ap.add_argument("--postura-rep", type=int, default=18, help="repetição do desvio de postura (padrão: 18)")
    args = ap.parse_args(argv)
    both = not args.figuras and not args.pacote
    if args.figuras or both:
        for p in figuras(args.amplitude_rep, args.postura_rep):
            print(f"figura: {p.relative_to(ROOT).as_posix()} ({p.stat().st_size / 1024:.0f} KB)")
    if args.pacote or both:
        out = pacote(args.amplitude_rep, args.postura_rep)
        n = sum(1 for p in out.rglob("*") if p.is_file())
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**20
        print(f"pacote: {out} ({n} arquivos, {size:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
