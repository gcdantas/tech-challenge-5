# 04 - Análise de áudio e texto clínico (Integrante 2)

Implementação local validada em 11/11/2026. **Azure real ainda não validado.**
Este capítulo é a contribuição do I2 para o relatório consolidado pelo I4.
Os comandos resumidos e o status ficam também na seção 8 do README.

## Escopo e arquitetura

O módulo recebe uma consulta WAV ou notas UTF-8. Usa Azure Speech para
transcrição e Azure AI Language para sentimento por frase e entidades. Regras
regex configuráveis identificam termos críticos e excluem negações e hipóteses
simples. O processamento de linguagem é o mesmo nas consultas e nas notas.
NumPy e Praat/Parselmouth extraem medidas vocais localmente.

```text
WAV -> validação PCM -> Speech contínuo/cache -> segmentos e palavras
                                                   |
TXT -> notas ordenadas por demo_offset ------------+-> Language/cache + regex
                                                   |        |
WAV + identificação do locutor -> medidas + z-score |        +-> Findings audio/text
                                                            +-> transcript.json / notes.json
```

Os objetos reutilizam `Finding`, `ModelInfo`, `read_findings` e `write_findings`.
Não houve mudança em contratos, `patients.yaml`, fusão, aplicação, vídeo ou
anomalias. O módulo não emite alertas. Não implementa diarização nem Text
Analytics for Health, que permanecem fora deste MVP.

| Arquivo | Responsabilidade |
|---|---|
| `modules/audio/speech.py` | WAV PCM16 mono/16 kHz, reconhecimento contínuo pt-BR e timestamps |
| `modules/audio/language.py` | SDK Language, sentimento por frase, NER e regex |
| `modules/audio/vocal.py` | Taxa, pausas, pitch, jitter/shimmer e comparação intra-locutor |
| `modules/audio/cache.py` | Cache JSON por hash e gravação JSON atômica |
| `modules/audio/config.py` | Validação do YAML e limiares |
| `modules/audio/pipeline.py` | Interface pública, notas, alinhamento e outputs |
| `modules/audio/run.py` | CLI de arquivo e cenário completo |
| `modules/audio/metrics.py` | WER, termos e concordância com referência humana |
| `modules/audio/fixtures/P001_mock.json` | Transcrição manual fictícia para execução offline |
| `config/audio.yaml` | Termos, parâmetros dos serviços e limiares experimentais |
| `tests/test_audio.py` | Testes offline específicos do I2 |

## Instalação e compatibilidade

Execute na raiz que contém `README.md`, `modules/` e `requirements.txt`.
Neste workspace ela é a subpasta `tech-challenge-5-main` dentro da pasta inicial.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

O ambiente desta entrega foi criado com `python -m uv venv --python 3.13 .venv`
porque o Python padrão era 3.11. Foi possível instalar **todas** as dependências
fixadas do projeto no CPython 3.13.18 / Windows x64. Dependências acrescentadas:

| Pacote | Versão validada | Uso |
|---|---|---|
| azure-cognitiveservices-speech | 1.53.0 | Speech SDK, incluindo inicialização nativa offline |
| azure-ai-textanalytics | 5.3.0 | Cliente síncrono TextAnalyticsClient; API 2023-04-01 |
| praat-parselmouth | 0.4.7 | Wheel CPython 3.13; pitch e comandos Praat |
| python-dotenv | 1.2.4 | Carregamento do `.env` local, sem sobrescrever o ambiente |

NumPy já pertencia ao projeto. Parselmouth e NumPy bastam para as cinco
características; não foi necessário adicionar librosa. Outros sistemas
operacionais não foram executados nesta validação.

## Recursos Azure e credenciais

Nenhum recurso foi provisionado, nenhuma chave foi criada e nenhuma chamada
de inferência Azure foi executada nesta entrega. Preparação humana:

1. No portal Azure, selecionar a assinatura e o grupo de recursos da equipe.
2. Criar ou reutilizar um recurso **Speech** e um recurso **Language** com
   análise de sentimento e reconhecimento de entidades. Selecionar região
   compatível e nível gratuito F0 quando disponível; conferir a oferta e as
   cotas na própria assinatura antes de confirmar qualquer custo.
3. Em **Keys and Endpoint**, copiar a chave e a região do Speech, e a chave e
   o endpoint HTTPS do Language para um `.env` local. As chaves são distintas.
4. Compartilhar credenciais apenas pelo canal privado acordado. Cada pessoa
   monta seu `.env`; nunca colocar esse arquivo ou as chaves no Git.
5. Autorizar explicitamente a execução real com `--allow-azure`.

```powershell
Copy-Item .env.example .env  # apenas se .env ainda não existir
```

Preencher `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, `AZURE_LANGUAGE_KEY` e
`AZURE_LANGUAGE_ENDPOINT`. Não é preciso preencher SMTP para executar o I2.
O módulo lê somente o `.env` da raiz, com `override=False`. A CLI exige
`--allow-azure` para cache misses. `--mock` nunca acessa a rede. Na interface
`analyze()`, a autorização equivalente é `TC_AUDIO_ALLOW_AZURE=1`.

Referências oficiais: [quickstart Speech](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-speech-to-text?pivots=programming-language-python),
[quickstart de sentimento](https://learn.microsoft.com/en-us/azure/ai-services/language-service/sentiment-opinion-mining/quickstart?pivots=programming-language-python)
e [TextAnalyticsClient](https://learn.microsoft.com/en-us/python/api/azure-ai-textanalytics/azure.ai.textanalytics.textanalyticsclient?view=azure-python).

## Preparação das gravações e baseline

Gravar 6 a 8 consultas **simuladas**, com consentimento dos participantes,
incluindo condição neutra e variações roteirizadas. Usar apenas a fala de um
locutor no WAV submetido ao indicador vocal; se houver entrevistador, preparar
uma faixa separada contendo a fala do participante. Não há separação automática
de locutores. Identidade e condição neutra dependem da conferência humana.

Entrada exigida: WAV PCM 16 bits, mono, 16000 Hz. Exemplo de conversão se FFmpeg
estiver instalado, sempre gerando outro arquivo:

```powershell
ffmpeg -i consulta_original.wav -ac 1 -ar 16000 -c:a pcm_s16le data/audio/P001_consulta.wav
```

Criar `data/audio/P001_consulta.wav.json` para habilitar o indicador:

```json
{"patient_id":"P001","speaker_id":"participante-01","single_speaker":true,"neutral_prefix_seconds":30}
```

Nesse caso, os primeiros 30 segundos devem ser neutros e contêm seis janelas
de 5 s; somente as janelas posteriores são avaliadas. Recomendam-se gravações
com pelo menos 60 s para a demonstração. O padrão exige cinco janelas válidas
de baseline para cada medida.

Alternativa: fornecer `--baseline data/audio/P001_baseline.wav`, gravado pelo
mesmo participante, com pelo menos 25 s válidos para os parâmetros padrão.
O arquivo `P001_baseline.wav.json` deve conter:

```json
{"patient_id":"P001","speaker_id":"participante-01","single_speaker":true,"condition":"neutral"}
```

Para baseline separado, a consulta não precisa declarar um prefixo neutro.
Os dois sidecars precisam confirmar o mesmo `patient_id` e `speaker_id`.
Não é permitido usar o mesmo arquivo como consulta e baseline separado.
Baseline externo também é transcrito, com cache, para medir sua taxa de fala.
Sem metadados do locutor, a linguagem ainda funciona, mas o vocal fica
`not_evaluated`. Identidades incompatíveis abortam antes das chamadas Azure.

## Execução e integração

### Demonstração local, sem credenciais

```powershell
python -m modules.audio.run --scenario --patient P001 --mock
```

O cenário usa uma fixture textual explicitamente fictícia e as duas notas
sintéticas. Gera os quatro arquivos nos caminhos acordados. Não cria um WAV
falso nem apresenta a fixture como resultado Speech. Não emite Finding vocal:
`vocal.status=not_evaluated`, motivo `mock_has_no_acoustic_evidence`.

Todos os Findings mock têm prefixo `[MOCK]`, modelo `mock-*` e
`evidence.mock=true`; os JSON também têm `mock=true`. Saídas reais existentes
não são substituídas por mocks. Para testar separadamente:

```powershell
python -m modules.audio.run --scenario --patient P001 --mock --outputs outputs/demo_i2
```

### Processamento real após autorização e preparação humana

```powershell
# Consulta e notas mapeadas no patients.yaml; quatro outputs
python -m modules.audio.run --scenario --patient P001 --allow-azure

# Mesmo cenário, baseline separado
python -m modules.audio.run --scenario --patient P001 --baseline data/audio/P001_baseline.wav --allow-azure

# Interface CLI exigida: apenas consulta + transcript.json
python -m modules.audio.run --input data/audio/P001_consulta.wav --patient P001 --out outputs/audio/P001/findings.jsonl --allow-azure

# Notas consecutivas + notes.json
python -m modules.audio.run --input data/notes --patient P001 --out outputs/text/P001/findings.jsonl --allow-azure

# Outro WAV usando explicitamente o offset da consulta do P001
python -m modules.audio.run --input data/audio/outra_consulta.wav --artifact consulta --patient P001 --out outputs/audio/P001/findings.jsonl --allow-azure

# Reutilizar cache sem autorizar novas chamadas: retirar --allow-azure
python -m modules.audio.run --scenario --patient P001
```

Se faltar cache, credencial, entrada ou mapeamento, o modo real falha com
mensagem explícita. Nunca recorre a mocks automaticamente. Use `--config`,
`--patients` e `--outputs` para caminhos alternativos. `TC_PATIENTS_PATH` e
`TC_OUTPUTS_DIR` também são respeitados. Uma pasta de notas deve conter apenas
notas mapeadas para o paciente selecionado. Texto de nota deve ter no máximo
5000 caracteres; não há truncamento silencioso.

```python
from modules.audio import analyze
findings = analyze("data/audio/P001_consulta.wav", "P001")
```

Essa interface retorna `list[Finding]`, usa o cache e aceita WAV, nota `.txt`
ou diretório de notas. Não escreve outputs; o integrador pode usar
`write_findings`. A CLI escreve Findings e JSON acompanhante.

### Contratos de saída

| Saída | Conteúdo |
|---|---|
| `outputs/audio/P001/findings.jsonl` | `source=audio`, termos, sentimento muito negativo e vocal quando mensurável |
| `outputs/audio/P001/transcript.json` | `segments` com start/end em segundos, timestamps, texto, palavras, sentimento, frases e `critical_terms` |
| `outputs/text/P001/findings.jsonl` | `source=text`, termos e sentimento quando sustentado |
| `outputs/text/P001/notes.json` | Notas ordenadas, texto sintético, timestamps, sentimento e comparação consecutiva |

`critical_terms` preserva `term`, `offset` e `length`, usados pelo dashboard;
offsets textuais são pontos de código Unicode. `evidence.term` preserva o termo
encontrado para a fusão; `canonical_term` agrupa variantes na comparação.
`excluded_mentions` explica quais menções foram negadas ou incertas.

As notas registram termos adicionados, ausentes na nota atual e persistentes
em `comparison`. “Ausente na nota atual” não afirma resolução clínica.
Os Findings dos termos também incluem `comparison`, `note`, `synthetic` e
`ausente_na_anterior`. Não se duplica o Finding para representar a mudança.

Timestamps são calculados exclusivamente com `contracts/patients.py`:

| Artefato | Offset preservado | Instante base |
|---|---:|---|
| consulta | +1800 s | 03/11/2026 10:30:00 -03:00 |
| nota_anterior | -86400 s | 02/11/2026 10:00:00 -03:00 |
| nota_recente | +2400 s | 03/11/2026 10:40:00 -03:00 |

Speech converte ticks de 100 ns para segundos; cada segmento soma seu tempo
interno ao offset. Sentimento é medido por frase, mas seu Finding usa o intervalo
do segmento que a contém; o offset da frase fica na evidência. Notas são eventos
pontuais. A transcrição de exemplo “falta de ar” em 75 s corresponde a 10:31:15.

## Modelos, regras e limitações

- Speech usa reconhecimento contínuo, `pt-BR`, saída detalhada e timestamps de
  palavras. A versão do SDK é registrada; o modelo Speech é gerenciado pelo serviço.
- Language usa `pt`, API `2023-04-01`, modelo solicitado `latest` configurável,
  logs do serviço desabilitados e offsets `UnicodeCodePoint`. As versões
  retornadas no corpo HTTP são registradas quando disponíveis; ausência é `null`.
- NER genérico não é um reconhecedor clínico. O dicionário regex define quais
  termos são críticos; entidades sobrepostas com confiança suficiente adicionam
  evidência `azure-ner`. O módulo não inventa entidade quando NER não a retorna.
- `sentimento_muito_negativo` exige frase textual verificável, classe negativa,
  confiança negativa >= 0,90 e pelo menos 8 caracteres. Confiança não equivale a
  gravidade clínica. Textos clínicos podem ser mal classificados por sentimento.
- Negação/hipótese são regras simples e conservadoras, não NLP clínico completo.
  Contextos históricos, terceiros, negação complexa e transcrições incorretas
  ainda precisam de revisão humana.
- Scores de termos são heurísticos configurados. A severidade deriva dos
  limites `medium=0,50`, `high=0,85`, `critical=0,98`; abaixo de 0,50 é `low`.
  Nenhum score representa probabilidade clínica calibrada.

### Indicador vocal

| Medida | Método e unidade |
|---|---|
| taxa_fala | Palavras/minuto pelos pontos médios dos timestamps de palavras Speech |
| pausas | Fração da janela em silêncio de pelo menos 0,20 s, RMS abaixo de -40 dBFS |
| pitch | Mediana do F0 de quadros vozeados, em Hz, Praat |
| jitter | Jitter local de períodos, fração, Praat |
| shimmer | Shimmer local de amplitude, fração, Praat |

Janelas não sobrepostas de 5 s. Exigem pelo menos 1 s vozeado. Sem timestamps de
palavras, a taxa fica `null`. Silêncio, sinal não vozeado, janela curta e medidas
indefinidas não são substituídos por valores fictícios. O método de pausa é
energético, não VAD treinado; ruído e microfone afetam o resultado.

Por característica: `z=(valor-media_baseline)/desvio_baseline`, desvio amostral
(`ddof=1`). Menos de cinco janelas válidas ou desvio <= 1e-6 impedem o cálculo
daquela medida. São necessários ao menos dois `abs(z)>=2,5` para emitir um
Finding. Score = menor desvio necessário para satisfazer esse número de medidas,
dividido por 5, limitado a 1. Baseline, medidas, estatísticas, z-scores e razões
de exclusão ficam no JSON/evidência. Parâmetros são ajustáveis no YAML.

As condições `not_evaluated`, `insufficient_data`, `disabled` e `evaluated`
distinguem falta de evidência de avaliação sem alteração. A confirmação de um
único locutor é humana. A comparação não usa valores populacionais. Limiares
não foram calibrados em gravações humanas. Jitter e shimmer em fala contínua
têm limitações; mudanças também podem resultar de ruído, intensidade ou microfone.
O resultado é **indicador experimental de alteração vocal, baseado em
características acústicas**. Não diagnostica disartria, fadiga clínica nem
dificuldade respiratória.

Referências: [sincronização Speech](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-speech-recognition-results?pivots=programming-language-python),
[Parselmouth](https://parselmouth.readthedocs.io/en/stable/examples.html)
e [definição de jitter local no Praat](https://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local____.html).

## Cache e privacidade

`outputs/audio/cache/azure-speech/` e `azure-language/` armazenam respostas
normalizadas. A chave SHA-256 inclui conteúdo, parâmetros, versão do SDK e
região/endpoint. Não inclui segredos. Arquivos idênticos com os mesmos parâmetros
reutilizam respostas; mudança no arquivo ou parâmetros invalida a chave.
Erros não são persistidos como sucesso. Mocks não ocupam namespaces Azure.

O cache pode conter texto transcrito, portanto deve ficar privado junto com
os áudios; `outputs/` já é ignorado pelo Git. Não há expiração automática.
`latest` pode mudar no servidor: para reavaliar deliberadamente, usar outra raiz
`--outputs` ou limpar somente o cache pretendido, e registrar a nova avaliação.
Para replay sem chave, manter a região e endpoint usados na criação do cache.
Resultados de Speech e Language são armazenados antes de aplicar as regras
locais, permitindo mudar limiares sem repetir requisições.

## Testes e evidências

```powershell
python -m pytest tests/test_audio.py -o addopts="" -q
python -m pytest -o addopts="" -q -ra
python -c "from contracts.finding import read_findings; print(len(read_findings('outputs/audio/P001/findings.jsonl')), len(read_findings('outputs/text/P001/findings.jsonl')))"
```

Validação local: **43 testes I2 passaram**. Suíte completa: **118 passaram e
6 foram pulados** por ausência dos pesos YOLOv8/OpenPose. Resultado registrado
também na seção 8.6 do README e em `outputs/audio/validation/pytest.xml`.
Testes usam eventos Speech simulados e objetos reais das classes de resposta
Language com cliente simulado, além de um teste do cliente Language real com
transporte HTTP offline para conferir requisições e deserialização. Não há rede
nesse teste. Testes acústicos executam Parselmouth de verdade
em tons artificiais, inclusive um desvio mensurável comparado ao baseline.
Esses sinais não representam consultas nem demonstram desempenho clínico.

O cenário mock gera 2 Findings de áudio (termo e sentimento) e 3 de texto.
A nota estável não gera termos afirmados; a recente inclui dispneia, sonolência
e falta de ar. Os arquivos passam no contrato Pydantic e nos leitores existentes.
A regra de fusão aceita `evidence.term` em teste com complementos simulados,
mas nenhum alerta real do cenário multimodal foi validado nesta entrega.

### Métricas: estado e reprodução

| Medida | Resultado disponível |
|---|---|
| WER de transcrição Azure | 0,23%: 1 diferença em 432 palavras revisadas, em 6 consultas |
| Precisão/recall/F1 de termos críticos em áudio | 100% / 100% / 100%: 6 TP, 0 FP e 0 FN nos documentos com termos |
| Concordância vocal com roteiro humano | 100% em 6/6 gravações, cobertura de 100% |
| Funcionamento local dos cálculos | Coberto por testes unitários e sinais artificiais, sem alegação clínica |

As medidas acima foram calculadas em 12/11/2026 com Azure real e uma referência
revisada por uma pessoa. O conjunto é sintético, contém um único locutor e foi
preparado para demonstrar os comportamentos esperados. Portanto, os resultados
não constituem validação independente nem permitem inferência de desempenho
clínico ou hospitalar. O resultado completo está em `outputs/audio/metrics.json`.

`metrics.py` calcula WER com distância de edição por palavras, preservando
acentos e removendo pontuação/capitalização; WER pode exceder 1. Termos são
avaliados como conjuntos canônicos por documento, com precisão/recall/F1 e
denominadores. Concordância vocal inclui cobertura: predições `null` são
não avaliadas, não acertos/erros. Para dados humanos revisados, criar um JSON
com a estrutura abaixo, substituir os campos de exemplo e só então marcar
`human_reviewed=true`:

```json
{
  "human_reviewed": false,
  "mock": false,
  "transcriptions": [{"id":"consulta-01","reference":"texto revisado","hypothesis":"texto do transcript.json"}],
  "terms": [{"id":"nota-01","source":"text","reference":[],"predicted":[]}],
  "vocal": {"reference":[true,false],"predicted":[null,null]}
}
```

```powershell
python -m modules.audio.metrics --reference data/audio/avaliacao_revisada.json --out outputs/audio/metrics.json
```

Guardar o gabarito e as gravações fora do Git. Separar métricas por modalidade
e por locutor/gravação; não usar o mesmo roteiro para calibrar e alegar avaliação
independente. Amostras pequenas simuladas não representam ambiente hospitalar.

## Pendências humanas e dos outros integrantes

**I2 / usuário:** recursos e credenciais Azure, execução real, consultas,
baseline, revisão humana, rótulos, métricas e evidências foram concluídos.
Resta preparar os slides da apresentação e revisar este capítulo no relatório.

**I4:** integração visual de `notes.json`, upload/chamada de `analyze()`, fusão
final, alertas/e-mail, relatório consolidado e validação multimodal real.
**I3:** vitais, prescrições e movimentação; **I1:** evidências e pendências de
vídeo já descritas em sua própria seção. Nenhuma dessas frentes foi implementada
ou substituída por este módulo. Ao gerar mocks dos outros integrantes, usar
`python -m scripts.make_mocks --skip i1 i2` para preservar vídeo e áudio/texto.

O MVP I2 está implementado, testado localmente e validado com Azure real em
amostra sintética revisada. As limitações da amostra permanecem explícitas.
