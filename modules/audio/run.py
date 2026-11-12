"""CLI de áudio, notas e cenário completo; modo real nunca faz fallback para mock."""
import argparse
import json
import sys

from azure.core.exceptions import AzureError

from .pipeline import AudioPipeline, resolve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="WAV PCM16/16k/mono, nota .txt, pasta de notas ou fixture --mock")
    parser.add_argument("--patient", required=True)
    parser.add_argument("--out", help="Arquivo findings.jsonl; documento acompanhante na mesma pasta")
    parser.add_argument("--scenario", action="store_true", help="Processa consulta e todas as notas mapeadas")
    parser.add_argument("--outputs", help="Raiz de outputs e cache (padrão outputs/ ou TC_OUTPUTS_DIR)")
    parser.add_argument("--artifact", help="Nome do artefato em patients.yaml quando a entrada tiver outro caminho")
    parser.add_argument("--baseline", help="WAV neutro do mesmo locutor, com metadados .wav.json")
    parser.add_argument("--config")
    parser.add_argument("--patients")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mock", action="store_true", help="Simulação explícita, nenhuma chamada Azure")
    modes.add_argument("--allow-azure", action="store_true", help="Autoriza chamadas Azure quando não há cache")
    args = parser.parse_args(argv)
    if not args.scenario and not (args.input and args.out):
        parser.error("Informe --input e --out, ou use --scenario")
    if args.scenario and (args.out or args.artifact):
        parser.error("--scenario usa --outputs e os artefatos de patients.yaml")
    try:
        pipeline = AudioPipeline(mock=args.mock, allow_azure=args.allow_azure, config_path=args.config,
                                 patients_path=args.patients, outputs=args.outputs)
        if args.scenario:
            summary = pipeline.scenario(args.patient, input=args.input, baseline=args.baseline)
        else:
            path = resolve(args.input)
            is_notes = path.is_dir() or path.suffix.lower() == ".txt"
            if is_notes and args.baseline:
                raise ValueError("--baseline só pode ser usado com áudio")
            result = (pipeline.notes(sorted(path.glob("*.txt")) if path.is_dir() else [path], args.patient,
                                     artifact_name=args.artifact) if is_notes else
                      pipeline.audio(path, args.patient, artifact_name=args.artifact, baseline=args.baseline))
            pipeline.save(result, args.out, "notes.json" if is_notes else "transcript.json")
            summary = {"findings": len(result.findings), "mock": pipeline.mock, "out": args.out}
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except AzureError:
        print("Erro: Azure Language falhou; confira credenciais, recurso, cota e conexão. Nenhum mock foi usado.", file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
