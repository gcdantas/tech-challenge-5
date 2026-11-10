# 3. Análise de vídeo

Responsável: Integrante 1 (I1). Código em `modules/video/`, limiares em
`config/video.yaml`.

> **Estado deste capítulo.**
> - **Já escrito:** método, modelos, regras e saídas, descrevendo o código já implementado.
> - **Resultados disponíveis:** o módulo foi testado em vídeos do dataset público REHAB24-6 (seções 3.2.2 e 3.10.3).
> - **Pendente:** os resultados com as gravações da equipe, que ainda não foram feitas. Esses trechos estão marcados como **A preencher** e não contêm números inventados.

## 3.1 Objetivo e escopo

O enunciado pede processar vídeos clínicos, apontar movimentos ou eventos fora
do padrão esperado com OpenPose (análise postural) e YOLOv8 (objetos e áreas
críticas) e gerar relatórios automáticos com os desvios encontrados. O módulo
atende isso em dois cenários curtos:

| Cenário | Para quê | O que o módulo produz |
|---|---|---|
| Sessão de fisioterapia (principal) | Apontar desvios na execução do exercício | Findings `source: video` dos tipos `amplitude_incompleta` e `postura_fora_padrao`, clipes e quadros anotados, relatório HTML |
| Paciente no leito | Alimentar a detecção de anomalias de movimentação (I3) e mostrar o YOLOv8 com área crítica | Série `movement.parquet` (~5 Hz) e quadros anotados com pessoa, leito e zona da borda. Nenhum Finding |

O cenário do leito não emite Findings. Saída do leito e imobilidade são
sinalizadas só pelo módulo de anomalias (`source: movement`), pela regra
"uma origem por evento" do contrato, que evita alertas duplicados. Detecção de
queda e vídeo de cirurgia estão fora do escopo do MVP (linha "Extra" do plano).

O módulo sinaliza desvios em relação a limiares configurados; não faz
diagnóstico. **O sistema sinaliza risco e apoia a equipe; a decisão é clínica.**

## 3.2 Dados

### 3.2.1 Gravações da equipe (dados da demonstração)

Os vídeos são gravados pela própria equipe, com consentimento por escrito de
quem aparece. Ficam em `data/video/`, fora do repositório. Cada vídeo é ligado
ao paciente sintético P001 em `data/patients.yaml`, com `kind: video_physio` ou
`kind: video_bed`. O paciente sintético P001 tem dados que vêm de fontes
distintas, associadas só para a demonstração técnica.

Plano de gravação (mínimo 6 vídeos, 8 se houver tempo; 20 a 60 s cada):

| Vídeo | Cenário | Roteiro | Duração | Resolução / FPS | Rótulos |
|---|---|---|---|---|---|
| A preencher | fisioterapia | exercício executado corretamente | | | |
| A preencher | fisioterapia | amplitude incompleta | | | |
| A preencher | fisioterapia | tronco fora da faixa | | | |
| A preencher | leito | paciente parado no leito | | | |
| A preencher | leito | paciente saindo do leito | | | |
| A preencher | | | | | |

> **A preencher:** exercício escolhido, posição da câmera (recomendado: de
> perfil, à altura do quadril, corpo inteiro em quadro), iluminação e quem
> rotulou os vídeos.

Recomendações de gravação que vêm das limitações do método (seções 3.10.3 e 3.12):
- **Câmera de perfil:** os ângulos são medidos em 2D, e a vista lateral reduz a distorção de perspectiva. No REHAB24-6, a vista em meio-perfil aumentou em até ~30° o ângulo medido do joelho em agachamentos corretos.
- **Paciente em destaque:** o módulo analisa a maior pessoa detectada. Se o fisioterapeuta aparecer, deve ficar mais longe da câmera que o paciente.
- **Leito inteiro em quadro:** o leito deve aparecer por completo e não ficar coberto a ponto de deixar de parecer um leito.

### 3.2.2 Dataset público para teste: REHAB24-6

Antes das gravações da equipe, o módulo foi testado em vídeos do **REHAB24-6**
(Černek, Sedmidubsky e Budikova, SISAP 2024; Zenodo, doi:10.5281/zenodo.13305826).

**Sobre o dataset**
- **Licença:** CC BY-NC 4.0, uso acadêmico e não comercial com citação.
- **Participantes:** 10 pessoas, que consentiram com a publicação.
- **Gravação:** cada pessoa executou 6 exercícios, filmados por duas câmeras a 30 FPS em ângulos ortogonais. Em cada vídeo a pessoa muda de direção no meio da série, então a mesma câmera a vê em duas vistas.
- **Anotações:** um fisioterapeuta orientou execuções corretas e incorretas. O dataset anota o início e o fim de cada repetição e se ela foi correta, mas **não o tipo do erro**.

**Vídeos usados:** 4 dos 136. Para cada exercício foi escolhida a câmera que vê o movimento no plano em que ele acontece.

| Vídeo | Exercício do dataset | Câmera | Vistas da pessoa | Duração | Repetições anotadas (incorretas) |
|---|---|---|---|---|---|
| PM_022 | Ex6, agachamento | 18 (vertical, 1080×1920) | perfil e meio-perfil | 92 s | 22 (1) |
| PM_043 | Ex6, agachamento | 18 | perfil e meio-perfil | 115 s | 20 (10) |
| PM_105 | Ex6, agachamento | 18 | perfil e meio-perfil | 110 s | 21 (11) |
| PM_001 | Ex1, abdução do braço direito | 17 (horizontal, 1920×1080) | frente | 101 s | 10 (5) |

**Como os dados foram usados**
- **Sem calibração:** os limiares do `config/video.yaml` **não foram ajustados** para esses vídeos. O objetivo era ver como o método se comporta com pessoas e vistas que ele não conhece, sem calibrar para outro cenário.
- **Abdução:** a única mudança foi uma configuração própria para o exercício, em `notebooks/rehab24/abducao.yaml`: ângulo do ombro, braço direito, ângulo que aumenta, braço ao menos na horizontal (80°) e tronco ereto (0–15°).
- **Escopo:** os vídeos não fazem parte dos dados do paciente sintético P001 nem da demonstração.

## 3.3 Visão geral do pipeline

```
                 data/patients.yaml (cenário + demo_offset)
                               │
vídeo .mp4 ──► leitura (OpenCV) ──┬─► Fisioterapia ─► YOLOv8-pose ─► ângulos ─► repetições ─► regras
                                  │                                                         │
                                  │                        Findings (source: video) ◄──────┤
                                  │                        clipes, quadros, report.html ◄──┘
                                  │
                                  ├─► Leito ─► YOLOv8 (pessoa, leito) + YOLOv8-pose (tronco)
                                  │                  └─► movement.parquet (~5 Hz) ─► I3
                                  │
                                  └─► OpenPose (OpenCV DNN) em N quadros ─► quadros com esqueleto
```

Interface comum do contrato:

```python
from modules.video import analyze
findings = analyze("data/video/P001_fisio.mp4", "P001")   # list[Finding]
```

```powershell
python -m modules.video.run --input data/video/P001_fisio.mp4 --patient P001 --out outputs/video/P001/findings.jsonl
```

## 3.4 Modelos

| Modelo | Uso | Implementação | Saída |
|---|---|---|---|
| YOLOv8n-pose | Pose da fisioterapia; inclinação do tronco no leito | `ultralytics` 8.4.208 (PyTorch) | 17 keypoints COCO por pessoa, com confiança |
| YOLOv8n | Pessoa e leito no cenário do leito | `ultralytics` 8.4.208, classes COCO `person` e `bed` | Caixas com confiança |
| OpenPose COCO (`pose_iter_440000`) | Requisito do enunciado: análise postural com OpenPose | Caffe, executado pelo módulo DNN do OpenCV 4.14 | 18 keypoints (mapas de calor) |

Os pesos são baixados por `python -m data.download_weights` para `weights/`,
fora do Git. Os do YOLOv8 vêm das releases oficiais da Ultralytics. O servidor
original da CMU com os pesos do OpenPose não responde, então o script usa um
espelho no Hugging Face e confere o SHA-256 do arquivo.

**Modelo usado na demo.** As regras da fisioterapia usam o YOLOv8-pose:
- roda com o mesmo framework do detector de objetos, então o pipeline tem uma só dependência de inferência;
- devolve keypoints por pessoa já associados a uma caixa;
- é rápido o bastante para analisar o vídeo inteiro a 15 quadros por segundo.

Velocidade medida na máquina de desenvolvimento, só com CPU (PyTorch sem GPU):

| Medição | Resultado |
|---|---|
| YOLOv8n-pose nos vídeos do REHAB24-6 (1080×1920 e 1920×1080) | 68 a 78 quadros por segundo de inferência |
| Vídeo de cerca de 100 s, analisado a 15 FPS e com o vídeo anotado gravado | 48 a 66 s de processamento |
| OpenPose no recorte da pessoa (REHAB24-6) | 0,84 FPS |
| OpenPose no quadro inteiro (foto 1280×720) | cerca de 5 s por quadro |

O OpenPose, no build do OpenCV distribuído pelo PyPI, roda só em CPU. Por isso
ele processa uma amostra de quadros, como evidência do requisito (seção 3.6).

> **A preencher (desejável, semana 4):** comparação OpenPose × YOLOv8-pose
> (seção 3.6.2) para fundamentar a escolha com números do próprio vídeo.

## 3.5 Cenário de fisioterapia

### 3.5.1 Ângulos articulares

Para cada quadro analisado, a pessoa principal é a de maior área de caixa
ponderada pela confiança. Keypoints com confiança abaixo de 0,3 contam como
ausentes. A partir dos 17 keypoints são calculados, por lado:

| Articulação | Ângulo no vértice | Pontos |
|---|---|---|
| Joelho | joelho | quadril – joelho – tornozelo |
| Quadril | quadril | ombro – quadril – joelho |
| Cotovelo | cotovelo | ombro – cotovelo – punho |
| Ombro | ombro | quadril – ombro – cotovelo |

Também se calcula a **inclinação do tronco**: o ângulo, em relação à vertical,
do vetor que vai do ponto médio dos quadris ao ponto médio dos ombros (0° =
ereto, 90° = horizontal).

Tratamento das séries:
- **Taxa de análise:** 15 quadros por segundo (`physio.process_fps`).
- **Lacunas:** buracos de keypoints de até 0,5 s são interpolados linearmente; lacunas maiores ficam vazias.
- **Suavização:** média móvel centrada de 0,3 s.
- **Lado do corpo:** escolhido automaticamente pela fração de quadros com o ângulo válido mais a confiança média dos keypoints daquele lado. Pode ser fixado em `primary_angle.side`.

### 3.5.2 Repetições e amplitude

O ângulo principal (padrão: joelho, com o agachamento como exercício) define as
repetições.

1. **Repouso:** o ângulo de repouso é estimado como o percentil 90 do ângulo no vídeo (percentil 10 quando o movimento aumenta o ângulo).
2. **Início e fim:** a repetição começa quando o ângulo se afasta mais de 15° do repouso e termina quando volta para dentro dessa faixa.
3. **Filtro de ruído:** só conta repetição com amplitude de pelo menos 25° e duração de pelo menos 0,8 s.
4. **Pico:** o extremo do ângulo dentro da repetição é o pico, que define a amplitude atingida.

A direção do movimento é configurável (`direction: decrease` para flexão, como
no agachamento, e `increase` para movimentos que abrem o ângulo, como a
abdução de ombro). Assim o mesmo código serve a outros exercícios trocando só
o `config/video.yaml`.

### 3.5.3 Regras de desvio

| Evento (`type`) | Quando é sinalizado | Faixa esperada (inicial) | Score | Severidade por desvio |
|---|---|---|---|---|
| `amplitude_incompleta` | O pico do ângulo principal na repetição não chega à faixa esperada | joelho até 100° (faixa 70–100°) | desvio ÷ 60°, limitado a 1 | < 15°: low · 15–35°: medium · ≥ 35°: high |
| `postura_fora_padrao` | A inclinação do tronco fica fora da faixa esperada por pelo menos 0,5 s seguidos dentro da repetição | 0–45° | desvio ÷ 40°, limitado a 1 | < 10°: low · 10–25°: medium · ≥ 25°: high |

- **Desvio:** a distância, em graus, entre o valor medido e o limite da faixa. O score fica normalizado em 0–1, como pede o contrato.
- **Unidade de avaliação:** as duas regras são avaliadas por repetição, e cada desvio vira um Finding com o número da repetição.
- **Postura fora das repetições:** não é avaliada.
- **Limiares:** os valores acima são **iniciais** e serão calibrados com os vídeos rotulados.

> **A preencher:** limiares finais após calibração e a justificativa de cada um
> (referência fisioterapêutica ou observação nos vídeos de execução correta).

### 3.5.4 Saída: Finding de vídeo

Exemplo gerado no teste automatizado com vídeo **sintético**. Ilustra o
formato, não é resultado.
- **Pose:** veio de um estimador sintético injetado (`fake-pose`). Com o modelo real, o campo `model` traz `yolov8n-pose` e a versão do `ultralytics`.
- **Caminhos:** encurtados aqui; dentro do repositório saem relativos à raiz.

```json
{"finding_id": "10971d4a-b875-4277-82e5-a4192421c6fd", "patient_id": "P001", "source": "video",
 "type": "amplitude_incompleta",
 "timestamp_start": "2026-11-03T10:10:06.266667-03:00", "timestamp_end": "2026-11-03T10:10:07.800000-03:00",
 "score": 0.508, "severity": "medium",
 "description": "Amplitude incompleta na repetição 3 (agachamento): ângulo do joelho chegou a 130°, esperado até 100°",
 "evidence": {"metric": "knee_angle", "value": 130.5, "expected": [70.0, 100.0], "deviation": 30.5,
              "rep": 3, "side": "left", "video": "…/P001_fisio.mp4",
              "media_start_s": 6.267, "media_end_s": 7.8},
 "model": {"name": "fake-pose", "version": "test"},
 "artifacts": ["…/video/P001/clip_P001_fisio_01.mp4",
               "…/video/P001/frames/P001_fisio/desvio_01_amplitude_incompleta.jpg"]}
```

Neste exemplo o `demo_offset` do vídeo era 600 s; por isso o desvio aos 6,3 s
do vídeo aparece às 10:10:06 na linha do tempo da demo.

## 3.6 OpenPose

### 3.6.1 Execução (requisito obrigatório)

```powershell
python -m modules.video.openpose --input data/video/P001_fisio.mp4 --patient P001 --max-frames 20
```

Como funciona:
- **Modelo:** OpenPose COCO (18 keypoints), carregado com `cv2.dnn.readNetFromCaffe`.
- **Entrada:** o quadro é redimensionado para altura 368, mantendo a proporção.
- **Localização dos pontos:** cada keypoint é o máximo do seu mapa de calor, aceito se a probabilidade passar de 0,1.
- **Amostra:** N quadros espalhados uniformemente pelo vídeo (padrão 20), por causa do custo em CPU.

Essa é uma abordagem de **pessoa única**: os campos de afinidade entre partes
(PAFs), que separam várias pessoas, não são usados. Em testes com duas pessoas
em cena, os pontos se misturaram entre elas. Para reduzir isso, por padrão o
YOLOv8 localiza a pessoa principal e o OpenPose roda só no recorte dela (com
15% de margem); `--no-crop` desliga o recorte. Com pessoas muito próximas ou
sobrepostas, o recorte pode ainda incluir partes da outra pessoa.

Saídas:
- `outputs/video/P001/frames/<video>/openpose_*.jpg`: quadros com o esqueleto;
- `outputs/video/P001/runs/openpose_<video>.json`: keypoints, tempo de inferência por quadro, FPS médio, proporção de keypoints ausentes e quadros sem nenhum keypoint;
- uma seção própria no `report.html`.

**Resultado no REHAB24-6** (PM_105, agachamento, 8 quadros espalhados pelo vídeo, com recorte da pessoa):
- 0,84 FPS;
- 11% de keypoints ausentes;
- nenhum quadro sem pose.

O esqueleto ficou coerente com o corpo nos quadros inspecionados.

> **A preencher:** 2 a 4 quadros do OpenPose no vídeo de fisioterapia da
> equipe (copiar para `docs/img/video/`), com FPS médio e proporção de
> keypoints ausentes.

### 3.6.2 Comparação OpenPose × YOLOv8-pose (desejável, semana 4)

Critérios definidos no plano:
- FPS médio;
- proporção de quadros com keypoints ausentes;
- variação quadro a quadro de um ponto estático;
- número de falhas completas de pose.

Os keypoints do OpenPose já ficam salvos no formato COCO-18, com conversão
para COCO-17 em `OpenPoseResult.to_coco17()`, para casar com o YOLOv8-pose nos
mesmos quadros.

> **A preencher se concluída:**
>
> | Critério | OpenPose (OpenCV DNN, CPU) | YOLOv8n-pose |
> |---|---|---|
> | FPS médio | | |
> | Quadros com keypoints ausentes | | |
> | Variação de ponto estático (px) | | |
> | Falhas completas | | |

## 3.7 Cenário do leito

### 3.7.1 Pessoa, leito e zona crítica

- **Pessoa e leito:** o YOLOv8n detecta as classes COCO `person` e `bed` a ~5 quadros por segundo (`bed.sample_hz`).
- **Caixa do leito:** é a mediana das caixas detectadas ao longo do vídeo, o que tolera oclusões momentâneas. Se o leito não for detectado em nenhum quadro, o módulo para com uma mensagem pedindo uma caixa manual (`bed.bed_box_override`, coordenadas normalizadas). Isso evita gerar uma série sem referência.
- **Zona crítica (borda do leito):** a faixa entre a caixa do leito e uma caixa interna recuada 12% de cada lado (`bed.edge_margin`). Nos quadros anotados ela aparece em laranja.
- **`in_bed`:** verdadeiro quando pelo menos 50% da área da caixa da pessoa está dentro da caixa do leito (`bed.in_bed_min_overlap`).

Cores nos quadros anotados:
- verde: pessoa no leito;
- laranja: pessoa no leito, com o centro na zona da borda;
- vermelho: pessoa fora do leito.

### 3.7.2 Série de movimentação (`movement.parquet`)

Arquivo `outputs/video/<paciente>/movement.parquet`, uma linha por amostra
(~5 Hz), com as colunas definidas no plano:

| Coluna | Tipo | Definição |
|---|---|---|
| `patient_id` | texto | paciente |
| `timestamp` | data/hora com fuso -03:00 | instante na linha do tempo da demo (seção 3.8) |
| `centroid_x`, `centroid_y` | real 0–1 | centro da caixa da pessoa, normalizado pela largura e altura do quadro |
| `torso_angle` | graus | inclinação ombros–quadris em relação à vertical, pelo YOLOv8-pose (0 em pé, ~90 deitado) |
| `velocidade` | unidades normalizadas/s | deslocamento do centroide por segundo, com média móvel de 0,6 s |
| `in_bed` | booleano | pessoa no leito (seção 3.7.1) |

- **Pessoa não detectada:** centroide, tronco e velocidade ficam vazios (NaN). `in_bed` mantém o último estado por até 1 s, para tolerar oclusão breve, e depois vale `false`.
- **Metadados do arquivo:** vídeo de origem, `demo_offset`, frequência efetiva, caixa do leito, unidades e modelos.
- **Rótulos de saída do leito** (`bed_exits` em `data/labels/<video>.yaml`) são copiados para `bed_exit_labels.json`, usado pelo I3 nas métricas de movimentação.

A série tem a duração do vídeo (20 a 60 s). A série estendida, que simula horas
de internação, é composta pelo simulador do I3 a partir deste arquivo e está
documentada no capítulo 5.

## 3.8 Alinhamento temporal

Os vídeos trazem o tempo da gravação, não o da demonstração. Para que os
Findings de todas as modalidades caiam na mesma janela de fusão de 30 minutos,
cada instante é convertido para a linha do tempo da demo:

```
instante = demo_reference + demo_offset do vídeo + tempo no vídeo
```

`demo_reference` e `demo_offset` vêm de `data/patients.yaml`. No P001 a
fisioterapia começa em 10:00:00 e o leito em 10:10:00, em 03/11/2026, fuso
-03:00. O mesmo cálculo vale para os Findings e para o `timestamp` do parquet.
Vídeos fora do `patients.yaml` exigem `--scenario` e usam `demo_offset` 0.

## 3.9 Relatório automático e artefatos

Cada execução atualiza `outputs/video/<paciente>/report.html`, com uma seção por
vídeo processado.

O que o relatório mostra:
- **Fisioterapia:**
  - resumo: exercício, duração, modelo, repetições, quantidade de desvios, quadros com pose e FPS;
  - tabela de desvios: horário na linha do tempo da demo e no vídeo, valor medido × faixa esperada, severidade, score, quadro anotado e link do clipe;
  - gráfico dos ângulos ao longo do vídeo, com os desvios destacados e os limites tracejados;
  - vídeo anotado e quadros de amostra.
- **OpenPose:** quadros com o esqueleto, FPS e proporção de keypoints ausentes.
- **Leito:** resumo, origem da caixa do leito, proporção do tempo no leito e quadros anotados.
- **Métricas:** a tabela, quando houver rótulos.

O relatório abre com o aviso de que o sistema sinaliza risco e apoia a equipe e
de que o paciente é sintético.

| Artefato | Descrição |
|---|---|
| `clip_<video>_<nn>.mp4` | clipe H.264 de cada desvio, com 1 s antes e depois, anotado |
| `annotated_<video>.mp4` | vídeo inteiro anotado (esqueleto, ângulos, repetição, desvio ativo) |
| `frames/<video>/desvio_*.jpg` | quadro no instante mais representativo de cada desvio |
| `frames/<video>/amostra_*.jpg`, `openpose_*.jpg`, `leito_*.jpg` | amostras, OpenPose e leito |
| `runs/<cenario>_<video>.json` | resumo da execução, base do relatório e das métricas |

## 3.10 Métricas e resultados

### 3.10.1 Método

- **Rótulos:** um arquivo por vídeo em `data/labels/<video>.yaml`, com início, fim e tipo de cada desvio observado. Modelo em `data/labels/video.example.yaml`.
- **Casamento:** um evento sinalizado conta como verdadeiro positivo quando tem o mesmo tipo de um rótulo e IoU temporal ≥ 0,3. O casamento é um para um, guloso por maior IoU.
- **Medidas:** precisão, recall e F1 por tipo de evento, por vídeo e no total.
- **FPS de inferência:** quadros analisados por segundo de modelo.
- **FPS de ponta a ponta:** quadros do vídeo cobertos por segundo de processamento.

```powershell
python -m modules.video.metrics --patient P001     # grava metrics.json e metrics.md
```

### 3.10.2 Resultados com as gravações da equipe

As métricas foram calculadas sobre uma amostra pequena, roteirizada e destinada
à validação técnica do protótipo; não representam validação clínica nem
desempenho esperado em ambiente hospitalar real.

> **A preencher após a gravação e a rotulagem:**
>
> | Evento | VP | FP | FN | Precisão | Recall | F1 |
> |---|---|---|---|---|---|---|
> | amplitude_incompleta | | | | | | |
> | postura_fora_padrao | | | | | | |
> | todos | | | | | | |
>
> | Vídeo | Cenário | FPS de inferência | FPS ponta a ponta | Hardware |
> |---|---|---|---|---|
> | | | | | |

### 3.10.3 Teste no dataset público REHAB24-6

As métricas foram calculadas sobre uma amostra pequena, roteirizada e destinada
à validação técnica do protótipo; não representam validação clínica nem
desempenho esperado em ambiente hospitalar real. São 4 vídeos de 4 pessoas,
processados com os limiares iniciais, sem calibração (seção 3.2.2).

**Como foi avaliado.** Como o dataset não diz o tipo do erro, a avaliação é por
repetição anotada:
- **Contagem de repetições:** cada repetição detectada é casada com uma anotada por IoU temporal ≥ 0,3, um para um.
- **Repetição sinalizada:** uma repetição anotada conta como sinalizada quando algum desvio do módulo (de qualquer tipo) cai dentro dela, com 0,25 s de tolerância, ou tem IoU ≥ 0,3 com ela.
- **Medidas:** precisão, recall e F1 da classe "repetição incorreta".

Script: `notebooks/rehab24/avaliacao.py`.

**Contagem de repetições e pose**

| Vídeo | Exercício | Repetições detectadas / anotadas / casadas | Quadros com pose | Desvios sinalizados |
|---|---|---|---|---|
| PM_022 | agachamento | 22 / 22 / 22 | 100% | 10 amplitude incompleta |
| PM_043 | agachamento | 20 / 20 / 20 | 100% | 10 amplitude incompleta |
| PM_105 | agachamento | 21 / 21 / 21 | 100% | 11 amplitude incompleta, 4 postura fora da faixa |
| PM_001 | abdução de ombro | 11 / 10 / 10 | 100% | nenhum |

- **Agachamento:** as 63 repetições anotadas foram encontradas, nas duas vistas e sem repetições a mais.
- **Abdução:** a repetição a mais é um levantamento do braço antes do trecho anotado.

**Sinalização de repetições incorretas**

| Exercício | Vista | VP | FP | FN | VN | Precisão | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| agachamento | perfil | 3 | 3 | 7 | 17 | 0,50 | 0,30 | 0,37 |
| agachamento | meio-perfil | 9 | 19 | 3 | 2 | 0,32 | 0,75 | 0,45 |
| agachamento | todas | 12 | 22 | 10 | 19 | 0,35 | 0,55 | 0,43 |
| abdução de ombro | frente | 0 | 0 | 5 | 5 | — | 0,00 | — |

**O que os resultados mostram**

A contagem de repetições é confiável. A sinalização de execução incorreta com
limiares absolutos não é. Os dados apontam três causas:

1. **Perspectiva.**
   - Nas repetições **corretas**, o pico do joelho ficou entre **72° e 111° de perfil** e entre **94° e 142° em meio-perfil**.
   - Com o limite fixo de 100°, quase todos os agachamentos corretos em meio-perfil foram sinalizados como amplitude incompleta (19 dos 22 falsos positivos).
   - É a limitação "medição 2D" da seção 3.12, agora medida.
2. **Variação entre pessoas.** Mesmo de perfil, um agachamento considerado correto pelo fisioterapeuta chegou a 111° (PM_043), acima do limite de 100°.
3. **Erros fora do que as duas regras cobrem.**
   - Nos vídeos de perfil, as repetições incorretas foram **mais profundas** que as corretas (pico de 60° a 83°): flexão excessiva, que a regra de amplitude incompleta não procura.
   - Na abdução, as repetições incorretas subiram um pouco menos (116°–123°) que as corretas (126°–132°). Isso fica muito acima do limite mínimo de 80°, então nenhuma foi sinalizada.

A regra de postura acertou o que se propõe: as 4 sinalizações no PM_105 são
repetições com o tronco inclinado à frente (48°–60°), todas anotadas como
incorretas.

### 3.10.4 Verificação técnica do software

Não são resultados de desempenho. São testes automatizados (`pytest`) que
conferem se o código faz o que este capítulo descreve:
- ângulos recuperados com exatidão em esqueletos sintéticos;
- contagem de repetições e as duas regras numa sessão sintética com uma repetição incompleta e uma com tronco inclinado;
- pipeline completo num vídeo sintético: Findings válidos, `demo_offset`, clipes, quadros e relatório;
- série do leito a ~5 Hz com a transição de `in_bed`;
- cálculo das métricas;
- rótulos desenhados nos quadros sem sobreposição, inclusive em vídeo vertical;
- CLI com o YOLOv8 real;
- OpenPose real numa foto de exemplo distribuída com a biblioteca `ultralytics`.

## 3.11 Exemplos

Os quadros anotados, clipes e o relatório da execução no REHAB24-6 são gerados
em `outputs/rehab24-6/video/P001/` (fora do Git), com o relatório em
`report.html`. Imagens do REHAB24-6 só podem entrar no relatório técnico com
a citação do dataset (licença CC BY-NC 4.0).

> **A preencher com prints dos vídeos da equipe (salvar em `docs/img/video/`):**
> 1. Quadro da fisioterapia com esqueleto do YOLOv8-pose e ângulos.
> 2. Quadro de um desvio de amplitude incompleta (com a tabela do relatório).
> 3. Quadro de um desvio de postura fora da faixa.
> 4. Quadros do OpenPose no mesmo vídeo.
> 5. Quadro do leito com pessoa, leito e zona da borda, antes e depois da saída.
> 6. Trecho do `report.html` e gráfico de ângulos.

## 3.12 Limitações

- **Medição 2D:** os ângulos são medidos na imagem, e a perspectiva da câmera altera os valores. No REHAB24-6, o mesmo agachamento correto mediu de 72° a 111° de perfil e de 94° a 142° em meio-perfil. Os limiares só valem para o enquadramento usado na calibração.
- **Limiares absolutos:** a mesma faixa vale para todas as pessoas, mas a amplitude correta varia de uma pessoa para outra.
- **Tipos de erro cobertos:** só amplitude incompleta e tronco fora da faixa. Flexão excessiva, que foi o erro mais comum no agachamento do REHAB24-6, não é sinalizada.
- **Pessoa analisada:** é a maior pessoa detectada. Com o fisioterapeuta mais perto da câmera que o paciente, a pessoa errada é analisada.
- **OpenPose de pessoa única:** pode misturar pessoas próximas, mesmo com o recorte. Roda em CPU, então cobre só uma amostra de quadros.
- **Limiares:** iniciais e definidos pela equipe. Não são critérios clínicos validados.
- **Postura:** avaliada só pela inclinação do tronco e só dentro das repetições. Outros desvios (por exemplo, joelho valgo) não são cobertos.
- **Leito:**
  - a detecção da classe COCO `bed` pode falhar com leitos hospitalares ou cobertos (há a caixa manual como alternativa);
  - `in_bed` usa sobreposição de caixas 2D, então uma pessoa à frente do leito pode ser contada como no leito.
- **Amostra:** poucos vídeos roteirizados, gravados pela equipe. As métricas não representam desempenho em ambiente hospitalar real.
- **Uso:** o sistema sinaliza risco e apoia a equipe; a decisão é clínica.

## 3.13 Próximos passos propostos

A partir do teste no REHAB24-6, em ordem de prioridade. Os dois últimos
precisam de decisão do grupo.

1. **Gravar os vídeos da equipe de perfil**, que é a vista em que as regras foram pensadas, e calibrar os limiares nessas gravações.
2. **Amplitude relativa à própria pessoa:** comparar cada repetição com a linha de base das primeiras repetições corretas do mesmo vídeo, no lugar de um limite absoluto. É a mesma ideia do indicador vocal do I2, que compara cada voz com ela mesma. Reduz o efeito da perspectiva e da variação entre pessoas, mas muda a semântica da regra.
3. **Flexão excessiva como terceiro tipo de evento.** Cobriria o erro mais comum do agachamento no REHAB24-6. Exige acrescentar um `type` à lista acordada no contrato.

## 3.14 Como reproduzir

Com as gravações da equipe:

```powershell
python -m pip install -r requirements.txt
python -m data.download_weights
python -m modules.video.run --input data/video/P001_fisio.mp4 --patient P001 --out outputs/video/P001/findings.jsonl
python -m modules.video.openpose --input data/video/P001_fisio.mp4 --patient P001 --max-frames 20
python -m modules.video.run --input data/video/P001_leito.mp4 --patient P001
python -m modules.video.metrics --patient P001
python -m pytest tests/test_video_*.py
```

Teste no REHAB24-6 (seção 3.10.3):

```powershell
python -m data.download_rehab24       # 4 vídeos (~104 MB) e anotações para data/video/rehab24-6/
$v = "data/video/rehab24-6"
python -m modules.video.run --patient P001 --scenario physio --outputs outputs/rehab24-6 --input "$v/PM_022-Camera18-30fps-transposed.mp4" "$v/PM_043-Camera18-30fps-transposed.mp4" "$v/PM_105-Camera18-30fps-transposed.mp4"
python -m modules.video.run --patient P001 --scenario physio --outputs outputs/rehab24-6 --config notebooks/rehab24/abducao.yaml --out outputs/rehab24-6/video/P001/findings_abducao.jsonl --input "$v/PM_001-Camera17-30fps.mp4"
python -m modules.video.openpose --patient P001 --outputs outputs/rehab24-6 --max-frames 8 --input "$v/PM_105-Camera18-30fps-transposed.mp4"
python -m notebooks.rehab24.avaliacao  # tabelas da seção 3.10.3
```

Referência do dataset: Černek, A., Sedmidubsky, J., Budikova, P.: REHAB24-6:
Physical Therapy Dataset for Analyzing Pose Estimation Methods. 17th
International Conference on Similarity Search and Applications (SISAP),
Springer, 2024. Dados: doi:10.5281/zenodo.13305826 (CC BY-NC 4.0).
