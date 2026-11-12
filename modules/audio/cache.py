"""Cache JSON por conteúdo/parâmetros; nunca armazena credenciais."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Any


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Arquivo temporário na mesma pasta: replace é atômico inclusive no Windows.
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as f:
            name = f.name
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


class ResponseCache:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def get_or_call(self, service: str, parameters: dict, call: Callable[[], Any]):
        key = digest(json.dumps({"schema": 1, "parameters": parameters},
                                sort_keys=True, ensure_ascii=False).encode("utf-8"))
        path = self.directory / service / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("key") != key:
                raise ValueError(f"Cache inválido: {path}")
            return record["response"]
        response = call()  # Falhas nunca são gravadas como sucesso.
        write_json(path, {"key": key, "response": response})
        return response
