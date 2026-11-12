"""Métricas sobre referências humanas fornecidas; nunca cria um gabarito."""
import argparse
import json
from pathlib import Path
import re
import unicodedata


def words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def wer(reference: str, hypothesis: str) -> dict:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        raise ValueError("WER exige referência não vazia")
    row = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        current = [i]
        for j, other in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (token != other)))
        row = current
    return {"wer": row[-1] / len(ref), "word_errors": row[-1], "reference_words": len(ref),
            "normalization": "NFC, minúsculas, sem pontuação; acentos preservados"}


def term_metrics(reference: list[str], predicted: list[str]) -> dict:
    """Conjuntos de termos canônicos por documento (não contagem de ocorrências)."""
    ref = {" ".join(words(s)) for s in reference}
    pred = {" ".join(words(s)) for s in predicted}
    tp, fp, fn = len(ref & pred), len(pred - ref), len(ref - pred)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def agreement(reference: list[bool], predicted: list[bool | None]) -> dict:
    if len(reference) != len(predicted) or not reference:
        raise ValueError("Concordância exige listas não vazias e de mesmo tamanho")
    if not all(type(v) is bool for v in reference) or not all(v is None or type(v) is bool for v in predicted):
        raise ValueError("Rótulos vocais devem ser booleanos; predição pode ser null (não avaliada)")
    pairs = [(r, p) for r, p in zip(reference, predicted) if p is not None]
    return {"agreement": sum(r == p for r, p in pairs) / len(pairs) if pairs else None,
            "evaluated": len(pairs), "total": len(reference), "coverage": len(pairs) / len(reference)}


def evaluate(manifest: dict) -> dict:
    if manifest.get("human_reviewed") is not True:
        raise ValueError("Declare human_reviewed=true somente após revisão humana do gabarito")
    output = {"human_reviewed": True, "mock": bool(manifest.get("mock", False)),
              "audio": [], "terms": [], "vocal": None}
    for sample in manifest.get("transcriptions", []):
        output["audio"].append({"id": sample["id"], **wer(sample["reference"], sample["hypothesis"])})
    if output["audio"]:
        errors = sum(s["word_errors"] for s in output["audio"])
        count = sum(s["reference_words"] for s in output["audio"])
        output["corpus_wer"] = errors / count
    for sample in manifest.get("terms", []):
        if sample["source"] not in ("audio", "text"):
            raise ValueError("Métrica de termos deve declarar source audio ou text")
        output["terms"].append({"id": sample["id"], "source": sample["source"],
                                **term_metrics(sample["reference"], sample["predicted"])})
    if manifest.get("vocal"):
        output["vocal"] = agreement(manifest["vocal"]["reference"], manifest["vocal"]["predicted"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="JSON revisado por uma pessoa")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from .cache import write_json
    result = evaluate(json.loads(Path(args.reference).read_text(encoding="utf-8")))
    write_json(Path(args.out), result)


if __name__ == "__main__":
    main()
