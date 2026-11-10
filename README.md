# Tech Challenge Fase 4: monitoramento multimodal de pacientes

Protótipo de demonstração técnica (pós IADT, FIAP), feito por um grupo de quatro
integrantes. Fusão tardia: cada modalidade (vídeo, áudio, texto clínico, sinais
vitais, prescrições, movimentação) é processada separadamente e convertida em
evidências estruturadas (Findings), combinadas por paciente e proximidade
temporal. **O sistema sinaliza risco e apoia a equipe; a decisão é clínica.**

O paciente sintético P001 tem dados que vêm de fontes distintas, associadas só
para a demonstração técnica.

> **Estado atual (10/11/2026):** marco M1 cumprido com mocks fictícios, exceto a
> decisão sobre a fonte das prescrições (I3). Módulo de vídeo (I1) implementado
> e testado em vídeos do dataset público REHAB24-6; ainda faltam as gravações da
> equipe e a calibração dos limiares. Contrato, `patients.yaml`,
> Streamlit e fusão são **versões provisórias feitas pelo I1** para destravar o
> trabalho; o dono é o I4. Áudio/texto (I2) e anomalias (I3) existem só como
> mocks. Detalhes em [Status](#status-por-integrante).

## Sumário

- [Responsabilidades](#responsabilidades)
- [Status por integrante](#status-por-integrante)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Contrato comum](#contrato-comum)
- [Mocks (M1)](#mocks-m1)
- [Aplicação, fusão e alertas](#aplicação-fusão-e-alertas)
- [Módulo de vídeo (I1)](#módulo-de-vídeo-i1)
- [Testes](#testes)
- [Linguagem obrigatória](#linguagem-obrigatória)
- [Dados e privacidade](#dados-e-privacidade)

## Responsabilidades

Fonte: plano do grupo, *Tech Challenge Fase 4 — Divisão do trabalho v5*.

| Integrante | Frente principal | Também responde por | Pasta | Capítulo do relatório |
|---|---|---|---|---|
| I1 | Análise de vídeo: pose, YOLOv8, OpenPose | `movement.parquet` para o I3; frames e clipes para o dashboard | `modules/video/` | `docs/03-video.md` |
| I2 | Áudio e texto clínico (Azure Speech + Azure AI Language) | Recursos Azure, chaves e cache das respostas | `modules/audio/` | `docs/04-audio.md` |
| I3 | Detecção de anomalias (vitais, prescrições, movimentação) | Simulador de vitais, prescrições e movimentação em fluxo | `modules/anomaly/` | `docs/05-anomalias.md` |
| I4 | Integração: contrato, aplicação, fusão, alertas, dashboard | Repositório, README, arquitetura, relatório consolidado, edição do vídeo | `contracts/`, `app/` | `docs/01`, `02`, `06` e PDF |

Regras que valem para todos:
- **Uma origem por evento.** `source: video` cobre só os desvios da
  fisioterapia (I1). `source: movement` cobre saída do leito e imobilidade (I3).
  `source: text` cobre as notas de evolução (I2).
- **Finding × alerta.** I1, I2 e I3 produzem Findings; só o I4 emite alertas,
  depois da fusão.
- **Mudança no schema** só com PR marcando os outros três integrantes.
- **Linha "Extra"** do plano só depois do M3.

## Status por integrante

Legenda: ✅ feito · 🟡 provisório ou parcial · ⬜ pendente

### Critérios do M1 (pipeline roda com mocks)

| Critério | Status |
|---|---|
| O Streamlit inicia | ✅ |
| `patients.yaml` carrega com os `demo_offset` | ✅ |
| Todos os mocks passam na validação Pydantic | ✅ (`tests/test_mocks_schema.py`) |
| Um Finding de cada origem, incluindo `text`, aparece na linha do tempo | ✅ (`tests/test_app_smoke.py`) |
| A fonte das prescrições está decidida | ⬜ (I3) |

### I1: vídeo

| Item | Status |
|---|---|
| Mock: `movement.parquet` e `findings.jsonl` com 2 eventos | ✅ |
| Pose com YOLOv8-pose, ângulos e amplitude por repetição | ✅ código; ✅ testado no REHAB24-6 (contagem de repetições: 63 de 63 no agachamento) |
| Regras dos 2 eventos com limiares em `config/video.yaml` | ✅ código; 🟡 testadas no REHAB24-6, F1 0,43 sem calibração; ⬜ calibrar nas gravações da equipe |
| OpenPose via OpenCV DNN com frames anotados (obrigatório) | ✅ código; 🟡 rodado no REHAB24-6; ⬜ rodar no vídeo da equipe |
| Teste em dataset público (REHAB24-6): download, execução e avaliação reproduzíveis | ✅ (`data/download_rehab24.py`, `notebooks/rehab24/`) |
| YOLOv8 pessoa + leito, zona da borda, `movement.parquet` com `in_bed` | ✅ código; ⬜ validar detecção do leito real |
| `demo_offset` nos Findings e no parquet | ✅ |
| Relatório automático HTML e clipes dos desvios | ✅ |
| Métricas (P/R/F1 por evento, FPS) | ✅ código; ⬜ números reais (dependem dos rótulos) |
| Gravar ≥ 6 vídeos (20–60 s) com consentimento por escrito | ⬜ |
| Rotular desvios e saídas do leito (`data/labels/`) e repassar ao I3 | ⬜ |
| `docs/03-video.md` | 🟡 método, modelos, regras, saídas e resultados no REHAB24-6 escritos; resultados com as gravações da equipe pendentes |
| Decidir com o grupo: amplitude relativa à própria pessoa e evento de flexão excessiva (capítulo 3, seção 3.13) | ⬜ |
| Desejável (semana 4): comparação OpenPose × YOLOv8-pose | ⬜ (keypoints já salvos para ela) |

### I2: áudio e texto clínico

| Item | Status |
|---|---|
| Mock: `transcript.json`, Findings de áudio e de texto | ✅ (provisório, feito pelo I1) |
| Recursos Azure Speech e AI Language, credenciais por canal privado, seção de ambiente no README | ⬜ |
| Gravar 6–8 consultas simuladas (normal, cansado, fala arrastada, ofegante) | ⬜ |
| Transcrição Azure Speech to Text (pt-BR) com timestamps | ⬜ |
| Sentimento e termos críticos (Azure AI Language + regex) | ⬜ |
| Notas de evolução sintéticas do P001 em `data/notes/`, Findings `source: text`, `notes.json` | ⬜ |
| Indicador experimental de alteração vocal (taxa de fala, pausas, pitch, jitter, shimmer; z-score contra a linha de base do próprio paciente) | ⬜ |
| Cache das respostas Azure em `outputs/audio/cache/` | ⬜ |
| `analyze()` e CLI em `modules/audio/` | ⬜ |
| Métricas (WER, acerto dos termos, concordância do indicador) e `docs/04-audio.md` | ⬜ |
| Desejável: Text Analytics for Health | ⬜ |

### I3: anomalias e simulador

| Item | Status |
|---|---|
| Mock: dessaturação, dose duplicada, saída do leito | ✅ (provisório, feito pelo I1) |
| Baixar MIMIC-IV Demo e/ou BIDMC (PhysioNet) e decidir a fonte das prescrições | ⬜ (critério do M1) |
| Simulador em fluxo (`config/simulator.yaml`) | ⬜ |
| Detectores com `update(ponto) -> list[Finding]`: vitais, prescrições, movimentação | ⬜ |
| Série de movimentação estendida a partir do `movement.parquet` do I1 | ⬜ |
| Anomalias injetadas, métricas, `series_flagged.parquet`, gráficos, `docs/05-anomalias.md` | ⬜ |
| Desejável: NEWS2 | ⬜ |

### I4: integração

| Item | Status |
|---|---|
| Contrato Finding (Pydantic + fixtures, incluindo `source: text`) | 🟡 provisório (I1); falta aprovação dos quatro |
| `patients.yaml` com `demo_offset` por artefato | 🟡 provisório (I1); formato a confirmar |
| Esqueleto do repositório | 🟡 provisório (I1) |
| Streamlit com linha do tempo | 🟡 esqueleto (I1) |
| Fusão: regra crítica do P001 e `alerts.jsonl` | 🟡 mínima (I1) |
| Combinação ponderada dos scores em `config/fusion.yaml` | ⬜ |
| Três níveis com deduplicação e cooldown por paciente e tipo | 🟡 cooldown só por regra |
| E-mail SMTP no alerta crítico (obrigatório), `config/alerts.yaml`, `email_sent_at`, contingência `.eml` | ⬜ |
| Upload de vídeo, áudio e nota no Streamlit; chamada a `analyze()` de I1 e I2; laço com o simulador do I3 | ⬜ |
| Dashboard com frames do I1, transcrição e notas do I2, séries do I3 | 🟡 transcrição e série do mock |
| Teste ponta a ponta do P001 com os módulos reais | ⬜ (hoje só com mocks) |
| `docs/01`, `02`, `06`, PDF consolidado, vídeo, submissão | ⬜ |
| Desejáveis: confirmação de leitura ("Ciente"), Azure Blob Storage, Docker Compose | ⬜ |

## Estrutura do repositório

```
contracts/            schema Finding, loader do patients.yaml (I4)
  fixtures/           um Finding de exemplo por origem
modules/video/        análise de vídeo (I1): fisioterapia, leito, OpenPose, relatório, métricas
modules/audio/        áudio e texto clínico (I2) - ainda vazio
modules/anomaly/      anomalias e simulador (I3) - ainda vazio
app/                  Streamlit, fusão e alertas (I4) - versão mínima
config/               video.yaml (limiares do I1), fusion.yaml (regras da fusão)
data/
  patients.yaml       pacientes sintéticos e demo_offset por artefato
  download_weights.py baixa os pesos dos modelos para weights/
  download_rehab24.py baixa vídeos de teste do dataset público REHAB24-6
  labels/             rótulos manuais dos vídeos (modelo: video.example.yaml)
  notes/              notas de evolução sintéticas (I2) - vazio
  video/              vídeos da equipe e de datasets públicos (fora do Git)
scripts/make_mocks.py gera os mocks fictícios do M1
notebooks/rehab24/    teste do módulo de vídeo no REHAB24-6 (config da abdução, avaliação)
docs/                 relatório técnico: 03-video.md (I1); enunciado oficial em PDF
tests/                pytest
outputs/              saídas geradas (fora do Git)
weights/              pesos dos modelos (fora do Git)
.env.example          modelo do .env (segredos)
```

## Instalação

Requer Python 3.13.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
copy .env.example .env                # Linux/macOS: cp .env.example .env
python -m data.download_weights       # pesos do vídeo (~220 MB), só se for rodar o módulo de vídeo
```

O `requirements.txt` fixa o OpenCV 4.x de propósito: a série 5 removeu o
importador Caffe usado pelo OpenPose.

### Variáveis de ambiente

| Variável | Onde | Uso |
|---|---|---|
| `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | `.env` | Azure Speech to Text (I2) |
| `AZURE_LANGUAGE_KEY`, `AZURE_LANGUAGE_ENDPOINT` | `.env` | Azure AI Language (I2) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL_FROM` | `.env` | e-mail do alerta crítico (I4) |
| `TC_OUTPUTS_DIR` | opcional | troca a pasta `outputs/` (usado nos testes) |
| `TC_PATIENTS_PATH` | opcional | troca o `data/patients.yaml` (usado nos testes) |

Os nomes das variáveis do `.env` são provisórios: I2 e I4 definem os finais.
Nenhum código usa o `.env` ainda.

### GPU (opcional)

O `requirements.txt` instala o PyTorch de CPU, que roda em qualquer máquina. Com
GPU NVIDIA, troque pelo build CUDA (confira em pytorch.org o índice certo para a
sua versão de CUDA; este comando não foi testado):

```powershell
python -m pip install --force-reinstall torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu128
```

## Contrato comum

### Finding (`contracts/finding.py`)

Toda saída de módulo que vira alerta é um `findings.jsonl` (um Finding por
linha):

```json
{"finding_id": "uuid4", "patient_id": "P001", "source": "video",
 "type": "postura_fora_padrao",
 "timestamp_start": "2026-11-03T10:15:00-03:00", "timestamp_end": "2026-11-03T10:15:12-03:00",
 "score": 0.87, "severity": "medium",
 "description": "Flexão de joelho 35° abaixo do esperado no exercício 2",
 "evidence": {"metric": "knee_angle", "value": 55, "expected": [85, 95]},
 "model": {"name": "yolov8n-pose", "version": "8.x"},
 "artifacts": ["outputs/video/P001/clip_0042.mp4"]}
```

- `source`: `video | audio | text | vitals | prescription | movement`.
- `severity`: `low | medium | high | critical`, derivada por limiares declarados em `config/<modulo>.yaml`.
- `score` normalizado em 0–1.
- Horários ISO 8601 com fuso `-03:00`; `timestamp_end` não pode ser anterior ao início.
- Campos extras são rejeitados; o que for específico de um módulo vai em `evidence`.

Funções úteis: `read_findings(path)`, `write_findings(path, findings)`.

### `type` usados até agora (a confirmar pelo grupo)

| `source` | `type` | Quem emite |
|---|---|---|
| `video` | `amplitude_incompleta`, `postura_fora_padrao` | I1 |
| `audio` | `termo_critico`, `indicador_vocal_alterado` | I2 |
| `text` | `termo_critico` | I2 |
| `vitals` | `dessaturacao` | I3 |
| `prescription` | `dose_duplicada` | I3 |
| `movement` | `saida_do_leito` | I3 |

Para termos críticos, a fusão lê o termo em `evidence.term`.

### `data/patients.yaml` (formato provisório)

```yaml
demo_reference: "2026-11-03T10:00:00-03:00"
patients:
  P001:
    description: ...
    artifacts:
      video_fisio: {kind: video_physio, path: data/video/P001_fisio.mp4, demo_offset: 0}
      consulta:    {kind: audio,        path: data/audio/P001_consulta.wav, demo_offset: 1800}
```

- `kind`: `video_physio | video_bed | audio | note | vitals | prescriptions`.
- `path` relativo à raiz; `null` enquanto não definido.
- `demo_offset` em segundos (pode ser negativo).
- Cada módulo grava `instante = demo_reference + demo_offset + tempo interno da mídia`
  (`PatientsFile.demo_time()`).

O P001 tem hoje: fisioterapia (0 s), leito (600 s), consulta (1800 s), nota
anterior (−86400 s), nota recente (2400 s), vitais e prescrições (0 s, `path: null`).

### Interface dos módulos

```python
analyze(input, patient_id) -> list[Finding]
```
```
python -m modules.<modulo>.run --input ... --patient P001 --out findings.jsonl
```

## Mocks (M1)

Os mocks deixam o pipeline rodar de ponta a ponta antes dos módulos reais. São
**todos fictícios** e gerados por um único script:

```powershell
python -m scripts.make_mocks              # gera tudo em outputs/
python -m scripts.make_mocks --skip i1    # não sobrescreve as saídas de um integrante (i1, i2, i3)
```

### Como um mock é marcado

- **Findings:** `description` começa com `[MOCK]`, `model.name` começa com
  `mock-` (`mock-i1`, `mock-i2`, `mock-i3`) e `evidence.mock = true`.
- **`movement.parquet`:** metadado de arquivo `mock=true`; as colunas seguem o
  plano sem alteração.
- **`transcript.json`:** campo `"mock": true`.
- **Streamlit:** mostra um aviso sempre que há Finding fictício na tela.

A saída é determinística: mesmos IDs e valores a cada execução.

### Linha do tempo do P001

Referência `2026-11-03T10:00-03:00`, com os `demo_offset` do `patients.yaml`.
Mapeia os passos do cenário P001 do plano:

| Horário | Origem | Evento | Severidade (score) | Passo do cenário |
|---|---|---|---|---|
| 10:00:12–10:00:18 | `video` | `amplitude_incompleta` (rep. 3, joelho 128°, esperado ≤ 100°) | medium (0,62) | 1 |
| 10:00:31–10:00:36 | `video` | `postura_fora_padrao` (rep. 6, tronco 52°, esperado ≤ 45°) | low (0,55) | — |
| 10:10:43–10:10:49 | `movement` | `saida_do_leito` | medium (0,70) | 2 |
| 10:20:00 | `prescription` | `dose_duplicada` (dipirona 1000 mg, 2×) | medium (0,70) | 5 |
| 10:30:30–10:31:30 | `audio` | `indicador_vocal_alterado` (z-scores em `evidence.zscores`) | medium (0,68) | 3 |
| 10:31:15–10:31:19 | `audio` | `termo_critico` "falta de ar" | medium (0,80) | 3 |
| 10:40:00 | `text` | `termo_critico` "dispneia aos esforços" (ausente na nota anterior) | medium (0,75) | 4 |
| 10:45:00–10:48:00 | `vitals` | `dessaturacao` (SpO2 87%, esperado 94–100%) | high (0,90) | 5 |

Os Findings da regra crítica (indicador vocal, termo respiratório e queda de
SpO2) ficam entre 10:30 e 10:45, dentro da janela de 30 min. A fusão emite o
alerta crítico às 10:48.

### Arquivos de cada mock

**I1: `outputs/video/P001/`**
- `findings.jsonl`: os dois eventos de fisioterapia da tabela.
- `movement.parquet`: 300 linhas a 5 Hz, de 10:10:00 a 10:10:59,8, com as
  colunas `patient_id`, `timestamp` (com fuso -03:00), `centroid_x`,
  `centroid_y`, `torso_angle`, `velocidade` e `in_bed` (bool).
  - O paciente fica deitado e parado no leito (tronco ~86° da vertical) até 40 s.
  - Depois senta, levanta e caminha para fora.
  - `in_bed` vira `false` em 10:10:43,6, o mesmo instante do `saida_do_leito` do mock do I3.

**I2: `outputs/audio/P001/` e `outputs/text/P001/`**
- `audio/P001/findings.jsonl`: `termo_critico` ("falta de ar", com o trecho em
  `evidence.segment_text`) e `indicador_vocal_alterado` (linha de base nos
  primeiros 30 s; z-scores de taxa de fala, pausas, pitch, jitter e shimmer).
- `audio/P001/transcript.json`, escrito à mão. Formato provisório (o I2 define o final):
  ```json
  {"mock": true, "aviso": "...", "patient_id": "P001", "audio": "data/audio/P001_consulta.wav",
   "language": "pt-BR",
   "segments": [{"start": 75.0, "end": 79.0,
                 "timestamp_start": "2026-11-03T10:31:15-03:00", "timestamp_end": "...",
                 "text": "Ontem à noite senti muita falta de ar ...", "sentiment": "negative",
                 "critical_terms": [{"term": "falta de ar", "offset": 26, "length": 11}]}]}
  ```
- `text/P001/findings.jsonl`: `termo_critico` "dispneia aos esforços" na nota
  recente (`evidence.note`, `evidence.ausente_na_anterior`).

**I3: `outputs/anomaly/P001/findings.jsonl`**
- `dessaturacao` (`vitals`), `dose_duplicada` (`prescription`) e
  `saida_do_leito` (`movement`).

### Como trocar um mock pelo módulo real

1. O módulo real grava **no mesmo caminho e no mesmo formato** do mock. O
   Streamlit e a fusão leem `outputs/<video|audio|text|anomaly>/<paciente>/findings.jsonl`
   sem saber se é mock.
2. A partir daí, regenere os mocks dos outros com `--skip`. Exemplo: depois de
   rodar o vídeo real, use `python -m scripts.make_mocks --skip i1`.
3. Rode `python -m pytest`. Os testes de mocks usam pasta temporária, mas o
   `read_findings()` valida qualquer saída real contra o contrato.

## Aplicação, fusão e alertas

```powershell
streamlit run app/streamlit_app.py
```

O que a versão mínima faz:
- Carrega o `patients.yaml` e lê todos os `findings.jsonl` em `outputs/`.
- Mostra, por paciente:
  - os artefatos com os `demo_offset`;
  - os alertas;
  - a linha do tempo multimodal e a tabela de Findings;
  - a série de movimentação;
  - a transcrição, com os termos críticos destacados.
- Roda a fusão de todos os pacientes e grava `outputs/alerts.jsonl`.

Fusão (`app/fusion.py`, regras em `config/fusion.yaml`):
- **Regra crítica do plano:** indicador vocal alterado + termo respiratório
  (na consulta ou na nota) + queda de SpO2 na mesma janela de 30 min. A lista
  de termos respiratórios fica no YAML.
- **Cooldown** de 30 min por regra e paciente.
- **Findings fora de regra** viram alerta isolado pela severidade:
  `low`/`medium` → atenção, `high` → urgente, `critical` → crítico.

Formato provisório de cada linha do `alerts.jsonl`:

```json
{"alert_id": "...", "patient_id": "P001", "level": "critico", "rule": "respiratorio_vocal_spo2",
 "description": "...", "finding_ids": ["...", "...", "..."],
 "window_start": "2026-11-03T10:30:30-03:00", "window_end": "2026-11-03T10:48:00-03:00",
 "created_at": "2026-11-03T10:48:00-03:00", "email_sent_at": null}
```

Ainda não implementado (I4):
- combinação ponderada dos scores;
- e-mail;
- confirmação de leitura;
- upload e chamada a `analyze()`;
- processamento incremental (hoje o `alerts.jsonl` é reescrito a cada execução).

## Módulo de vídeo (I1)

Os vídeos da equipe ficam em `data/video/` (fora do Git). Cada um é ligado ao
paciente no `data/patients.yaml` (`kind: video_physio` ou `video_bed`, com o
`demo_offset`). Os pesos vão para `weights/` (fora do Git):

```powershell
python -m data.download_weights           # YOLOv8n-pose, YOLOv8n e OpenPose COCO (~220 MB)
```

O servidor da CMU com os pesos do OpenPose está fora do ar. O script usa o
espelho no Hugging Face e confere o SHA-256.

### Fisioterapia

YOLOv8-pose → ângulos → regras. Emite Findings `source: video`
(`amplitude_incompleta`, `postura_fora_padrao`), clipes e quadros anotados e o
relatório.

```powershell
python -m modules.video.run --input data/video/P001_fisio.mp4 --patient P001 --out outputs/video/P001/findings.jsonl
```

- **Exercício:** o padrão é agachamento, com o ângulo do joelho como ângulo
  principal. Para outro exercício, troque `physio.primary_angle` em
  `config/video.yaml` (`joint`: knee, hip, elbow ou shoulder; `direction`:
  decrease ou increase).
- **Pessoa analisada:** a de maior caixa detectada. Com o fisioterapeuta em
  cena, enquadre o paciente mais perto da câmera.

### Leito

YOLOv8 detecta pessoa e leito; a borda do leito é a zona crítica. O módulo gera
só o `outputs/video/P001/movement.parquet` (~5 Hz) e os quadros anotados.
- **Sem Findings:** saída do leito e imobilidade são do I3. Rodar só o leito não
  altera o `findings.jsonl`.
- **Leito não detectado:** defina `bed.bed_box_override` em `config/video.yaml`.

```powershell
python -m modules.video.run --input data/video/P001_leito.mp4 --patient P001
```

Unidades do parquet (a confirmar com o I3):
- `centroid_x` e `centroid_y`: centro da caixa da pessoa, normalizado 0–1.
- `torso_angle`: graus em relação à vertical (0 em pé, 90 deitado).
- `velocidade`: unidades normalizadas por segundo.
- **Pessoa fora de vista:** o último `in_bed` vale por até 1 s; depois fica
  `false`, com centroide NaN.

Vídeo fora do `patients.yaml`: acrescente `--scenario physio` ou `--scenario bed`
(o `demo_offset` vira 0). Vários vídeos podem ir no mesmo `--input`.

### OpenPose

Obrigatório pelo enunciado. Roda pelo OpenCV DNN, com o modelo COCO, em N
quadros espalhados pelo vídeo. Por padrão, o YOLOv8 recorta a pessoa principal
antes (`--no-crop` desliga). Em CPU leva cerca de 5 s por quadro em 720p.

```powershell
python -m modules.video.openpose --input data/video/P001_fisio.mp4 --patient P001 --max-frames 20
```

### Métricas

Precisão, recall e F1 por evento contra rótulos manuais, mais o FPS.
1. Copie `data/labels/video.example.yaml` para `data/labels/<nome_do_video>.yaml`.
2. Anote início, fim e tipo de cada desvio, em segundos do vídeo.
3. Para vídeos do leito, anote os instantes de saída em `bed_exits`; eles viram
   `bed_exit_labels.json` para o I3.
4. Rode:

```powershell
python -m modules.video.metrics --patient P001
```

### Saídas em `outputs/video/<paciente>/`

| Arquivo | Conteúdo |
|---|---|
| `findings.jsonl` | Findings da fisioterapia (para o I4) |
| `movement.parquet` | série do leito (para o I3) |
| `bed_exit_labels.json` | rótulos de saída do leito (para o I3), se houver |
| `report.html` | relatório automático: resumo, horários dos desvios na linha do tempo da demo, gráfico de ângulos, quadros anotados, OpenPose, leito, métricas |
| `clip_<video>_<nn>.mp4` | clipe anotado de cada desvio (H.264, com 1 s de folga) |
| `annotated_<video>.mp4` | vídeo inteiro anotado (desligue com `--no-annotated-video`) |
| `frames/<video>/` | quadros anotados (desvios, amostras, OpenPose, leito) |
| `runs/<cenario>_<video>.json` | resumo de cada execução (usado pelo relatório e pelas métricas) |
| `metrics.json`, `metrics.md` | tabela de métricas |

Os limiares de `config/video.yaml` são valores iniciais e precisam ser
calibrados com os vídeos reais. O relatório sinaliza desvios em relação a esses
limiares; não é diagnóstico.

### Teste com o dataset público REHAB24-6

Antes das gravações da equipe, o módulo foi testado em 4 vídeos do REHAB24-6
(Zenodo, doi:10.5281/zenodo.13305826, licença CC BY-NC 4.0, uso acadêmico com
citação). São 3 agachamentos e 1 abdução de braço, com repetições anotadas por
um fisioterapeuta como corretas ou incorretas.

```powershell
python -m data.download_rehab24       # 4 vídeos (~104 MB) e anotações; lê só esses trechos do zip de 2,5 GB
python -m notebooks.rehab24.avaliacao  # tabelas, depois de rodar o módulo (comandos no capítulo 3, seção 3.14)
```

Resumo (sem calibrar os limiares):
- **Contagem de repetições:** 63 de 63 no agachamento.
- **Sinalização de repetições incorretas:** F1 de 0,43.
- **Causa da sinalização fraca:** os ângulos mudam com a vista da câmera (perfil × meio-perfil), e os erros do dataset são principalmente de flexão excessiva, que as regras não cobrem.

Análise completa e próximos passos em `docs/03-video.md` (seções 3.10.3 e 3.13).
As métricas foram calculadas sobre uma amostra pequena, roteirizada e destinada
à validação técnica do protótipo; não representam validação clínica nem
desempenho esperado em ambiente hospitalar real.

## Testes

```powershell
python -m pytest
```

| Arquivo | O que cobre |
|---|---|
| `test_contract.py` | regras do Finding e leitura/escrita do JSONL |
| `test_patients.py` | `patients.yaml` e alinhamento temporal |
| `test_mocks_schema.py` | todos os mocks validam, estão nos caminhos do plano, são marcados e determinísticos |
| `test_p001_smoke.py` | cenário P001: alerta crítico pela regra, controle negativo fora da janela |
| `test_app_smoke.py` | Streamlit inicia com e sem saídas |
| `test_video_*.py` | geometria, regras, pipeline, CLI, OpenPose, leito, métricas e desenho dos rótulos do vídeo |

Os testes de vídeo usam vídeos sintéticos gerados em pasta temporária, nunca
dados da demo. Os que rodam modelos reais (YOLOv8, OpenPose) são pulados se os
pesos não tiverem sido baixados. O teste do OpenPose usa uma foto de exemplo
que vem dentro do pacote `ultralytics`.

## Linguagem obrigatória

Vale para relatório, vídeo, dashboard e relatórios gerados (plano v5):

| Em vez de | Escrever |
|---|---|
| "O sistema diagnostica" | "O sistema sinaliza risco e apoia a equipe; a decisão é clínica" |
| "Detecta disartria" / "detecta dificuldade respiratória" | "Indicador experimental de alteração vocal, baseado em características acústicas" |
| "O paciente P001" | "O paciente sintético P001, cujos dados vêm de fontes distintas associadas só para a demonstração técnica" |
| "Tempo real" | "Tempo quase real em ambiente de demonstração, com reprodução acelerada de dados históricos e emissão incremental de alertas" |
| "Horas de movimentação do paciente" | "Série de movimentação composta a partir de um vídeo curto, estendida de forma documentada para simular a internação" |
| "Risco calculado pela fusão" | "A combinação ponderada é uma heurística de priorização do protótipo e não representa probabilidade clínica calibrada; regras explícitas têm precedência" |
| "O modelo tem F1 de 0,9" (sem contexto) | "As métricas foram calculadas sobre uma amostra pequena, roteirizada e destinada à validação técnica do protótipo; não representam validação clínica nem desempenho esperado em ambiente hospitalar real" |

## Dados e privacidade

- **Fora do Git** (veja `.gitignore`): dados brutos, pesos de modelo, saídas e
  segredos (`.env`).
- **Gravações da equipe:** vídeos e áudios ficam em `data/video/` e
  `data/audio/`, gravados com consentimento por escrito.
- **E-mails de alerta:** só para endereços da equipe.
- **Fontes de dados:** só dados públicos, sintéticos ou da equipe.
- **Datasets públicos:** respeitam a licença de cada um. O REHAB24-6 (CC BY-NC
  4.0) só é usado para testes e, se aparecer no relatório, com a citação
  (Černek, Sedmidubsky e Budikova, SISAP 2024). Os vídeos ficam em
  `data/video/rehab24-6/`, fora do Git.
