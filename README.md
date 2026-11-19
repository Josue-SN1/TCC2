# Experimento Manchester T1D-UOM

Este projeto reproduz uma análise exploratória para estimar um proxy ordinal
de risco de cetoacidose diabética (DKA) em pessoas com diabetes tipo 1.
A análise usa dados multimodais da base T1D-UOM/Universidade de Manchester.

Este repositório não contém a base bruta, resultados processados, PDFs,
arquivos compactados ou arquivos processados. O código, o texto-fonte do volume
em `volumefinal/` e o script de figuras são versionados. A base pública, os PDFs
finais, os relatórios JSON e as PNG geradas são locais e podem ser regenerados
pelos scripts descritos abaixo.

## Separação metodológica

- Modelos globais são treinados com janelas de todos os participantes.
- O participante é usado como unidade de cobertura, auditoria e avaliação:
  não são treinados modelos exclusivos para cada indivíduo.
- Janelas de um mesmo participante não são misturadas aleatoriamente entre
  treino e teste.
- A divisão temporal usa os primeiros 70% das janelas válidas de cada
  participante para treinamento e os 30% finais para teste.
- A validação Leave-One-Participant-Out (LOPO) deixa cada participante
  integralmente fora do treinamento em sua rodada.
- O alvo é um `risk_score` ordinal de 0 a 5 derivado de glicose e ausência do
  sensor. Ele não representa diagnóstico clínico de DKA.

## Interpretação das classes do proxy

O `risk_score` soma cinco critérios de hiperglicemia e ausência do sensor.
Assim, as classes representam o número de critérios atendidos em uma janela:

- **Classe 0:** nenhum critério atendido;
- **Classe 1:** exatamente um critério atendido;
- **Classe 2:** exatamente dois critérios atendidos;
- **Classe 3:** exatamente três critérios atendidos;
- **Classe 4:** exatamente quatro critérios atendidos;
- **Classe 5:** todos os cinco critérios atendidos.

Na análise classificatória binária, a classe 0 é negativa e as classes 1 a 5
são positivas. As classes são ordinais e exploratórias; não correspondem a
diagnósticos clínicos nem a etapas clínicas sequenciais de DKA.

O limiar `risk_score >= 1` gera desbalanceamento em favor da classe positiva:
no horizonte de 7 dias, há aproximadamente 74% de janelas positivas no teste
temporal e no LOPO. Por isso, precisão, recall, acurácia balanceada e ROC-AUC
devem ser analisados juntos, e não apenas a acurácia. Não há vazamento temporal
direto entre janela `t` e alvo `t+1`, mas o rótulo é derivado da própria série
de glicose usada pelos atributos; isso pode produzir associação mais forte sem
validar DKA clínica.

## Base de dados

Baixe separadamente o conjunto
[T1D-UOM no Zenodo](https://doi.org/10.5281/zenodo.15169263), respeitando sua
licença e condições de uso. Extraia a pasta principal na raiz deste projeto
com o nome:

```text
ManchesterCSCoordinatedDiabetesStudy-main/
```

A base contém glicose, sono, atividade física, insulina, nutrição e dados
demográficos. Os experimentos deste repositório usam somente as modalidades
explicitadas em cada comando.

## Instalação

Recomenda-se Python 3.12, pois os resultados foram obtidos e validados nesse
intérprete. Clone o repositório e execute os comandos a partir da raiz:

```powershell
git clone https://github.com/Josue-SN1/TCC2.git
cd TCC2
```

Execute os comandos a partir da raiz do projeto.

Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reprodução

### Reprodução mínima completa

Após obter a base Manchester, baixar as dependências e executar o
pré-processamento, esta sequência reproduz a análise principal de 7 dias:

```powershell
python inspect_manchester_data.py
python preprocess_manchester.py
python audit_manchester_alignment.py
python audit_temporal_windows.py
python train_manchester_baseline.py --days 7 --positive-threshold 1
python validate_manchester_lopo.py --days 7 --positive-threshold 1
python generate_analysis_figures.py
```

No horizonte principal, espera-se 150 janelas de treino e 78 de teste na
divisão temporal 70/30 e 228 janelas no LOPO. Os resultados são escritos em
`Manchester_processed/`. Esse diretório e as PNG em `figures/` e
`volumefinal/figuras/` são artefatos locais ignorados pelo Git.

### Incerteza e análise de erros

Com o relatório LOPO de 7 dias já gerado, execute:

```powershell
python analyze_lopo_uncertainty.py --bootstrap-samples 2000 --seed 42
```

O comando reexecuta as 17 rodadas LOPO com os hiperparâmetros fixados,
aplica bootstrap por participante e gera em `Manchester_processed/`:

- `lopo_uncertainty_7d.json`;
- `lopo_uncertainty_7d.md`;
- `lopo_uncertainty_7d.png`;
- `lopo_participant_error_7d.png`.

### 1. Inventário da base

```bash
python inspect_manchester_data.py
```

Gera `manchester_inventory.json` e `manchester_inventory.md`, que registram a
estrutura da base sem treinar modelos.

### 2. Pré-processamento

```bash
python preprocess_manchester.py
```

Gera a grade processada em `Manchester_processed/`. A glicose é mantida em
mmol/L e também convertida para mg/dL. Os dados brutos não são modificados.

### 3. Auditorias

```bash
python audit_manchester_alignment.py
python audit_temporal_windows.py
```

A primeira auditoria verifica alinhamento e cobertura multimodal. A segunda
verifica ordem cronológica, duplicatas e integridade das janelas.

### 4. Baseline temporal

```bash
python train_manchester_baseline.py --days 7 --positive-threshold 1
```

Treina XGBoost, Random Forest e Ridge com os primeiros 70% das janelas por
participante e avalia os 30% finais. Também pode incluir modalidades:

```bash
python train_manchester_baseline.py --days 7 --include-sleep
python train_manchester_baseline.py --days 7 --include-sleep-summary
python train_manchester_baseline.py --days 7 --include-nutrition
python train_manchester_baseline.py --days 7 --include-basal
python train_manchester_baseline.py --days 7 --include-bolus
python train_manchester_baseline.py --days 7 --include-basal --include-bolus
```

### 5. Validação LOPO

```bash
python validate_manchester_lopo.py --days 7 --positive-threshold 1
```

Para produzir um relatório com nome explícito:

```bash
python validate_manchester_lopo.py --days 7 --include-sleep-summary \
  --output Manchester_processed/lopo_report_7d_sleep_summary.json
```

As doses também podem ser avaliadas separadamente no LOPO:

```bash
python validate_manchester_lopo.py --days 7 --include-basal \
  --output Manchester_processed/lopo_7d_glucose_basal.json
python validate_manchester_lopo.py --days 7 --include-bolus \
  --output Manchester_processed/lopo_7d_glucose_bolus.json
python validate_manchester_lopo.py --days 7 --include-basal --include-bolus \
  --output Manchester_processed/lopo_7d_glucose_basal_bolus.json
```

Os atributos usam o total de `basal_basal_dose` ou `bolus_bolus_dose`
acumulado dentro da janela `t` para prever o `risk_score` da janela `t+1`.
Eles não devem ser interpretados como confirmação de dose esquecida ou dose
incorreta; não há desfecho clínico observado que permita essa inferência.

No Windows, use barras normais ou escreva o comando em uma única linha.

### Teste comparativo com risco positivo mais restritivo

Para definir a classe positiva como `risk_score >= 3`, mantenha os mesmos
dados e modelos e repita separadamente o baseline e o LOPO:

```bash
python train_manchester_baseline.py --days 7 --positive-threshold 3 \
  --output Manchester_processed/baseline_threshold3_7d.json
python validate_manchester_lopo.py --days 3 --positive-threshold 3 \
  --output Manchester_processed/lopo_threshold3_3d.json
python validate_manchester_lopo.py --days 7 --positive-threshold 3 \
  --output Manchester_processed/lopo_threshold3_7d.json
python validate_manchester_lopo.py --days 14 --positive-threshold 3 \
  --output Manchester_processed/lopo_threshold3_14d.json
```

Os resultados devem ser comparados separadamente do limiar `risk_score >= 1`.
Eles reduzem a prevalência positiva e produzem novas matrizes de confusão,
precisão, recall, acurácia balanceada e ROC-AUC.

### 6. Busca de combinações de atributos

```bash
python search_feature_combinations.py --days 7
python search_feature_combinations.py --days 7 --nutrition-observed-only
```

O primeiro comando inclui os participantes disponíveis e mantém a glicose em
todas as combinações. O segundo retém somente participantes com registros
nutricionais observados.

### 7. Figuras

Depois de gerar os relatórios JSON, execute:

```bash
python generate_analysis_figures.py
```

O script cria PNG locais em `figures/`. Para compor o volume em
`volumefinal/`, copie os arquivos PNG gerados para
`volumefinal/figuras/`. Eles não são enviados ao Git e podem ser regenerados
a partir dos dados e relatórios locais.

## Modelos

- **XGBoost:** `XGBRegressor`, 100 estimadores, profundidade máxima 3,
  taxa de aprendizado 0,05, função objetivo `reg:squarederror`, semente 42.
- **Random Forest:** 300 árvores, mínimo de duas amostras por folha,
  semente 42.
- **Ridge:** padronização por `StandardScaler` seguida de
  `Ridge(alpha=1.0)`.

Os hiperparâmetros foram mantidos fixos nesta reprodução. Ridge é um baseline
linear regularizado; Random Forest e XGBoost representam relações não lineares.

## Horizonte, atributos e alvo

Os scripts podem construir janelas de 3, 7 ou 14 dias:

```bash
python validate_manchester_lopo.py --days 3
python validate_manchester_lopo.py --days 7
python validate_manchester_lopo.py --days 14
```

Atributos de glicose: média, desvio padrão, mínimo, máximo e fração ausente.

Atributos adicionais:

- `sleep_ts`: passos, frequência cardíaca e estresse estimado;
- resumo diário de sono: duração e tempos em estágio estimado;
- nutrição: carboidratos, proteínas, gorduras e fibras.
- insulina: totais de dose basal e bolus agregados por janela.

O escopo multimodal foi limitado às combinações realmente executadas: glicose
isolada, glicose com resumo diário de sono, glicose com sinais pontuais de
`sleep_ts`, glicose com nutrição e combinações dessas modalidades no LOPO.
Não foram treinados modelos exclusivos por participante.

O `risk_score` soma cinco critérios: três associados à hiperglicemia e dois à
ausência do sensor. Ele não usa cetonas, pH, bicarbonato, hospitalização ou
diagnóstico clínico.

## Métricas

A regressão reporta MAE, RMSE e R². A análise classificatória é secundária:
transforma o `risk_score` em classe binária com `risk_score >= 1` e reporta
precisão, recall, F1, acurácia balanceada e ROC-AUC.

Essas métricas avaliam o proxy definido nesta pesquisa. Não devem ser
interpretadas como validação clínica de DKA.

## Arquivos principais

- `inspect_manchester_data.py`: inventário da base.
- `preprocess_manchester.py`: normalização e criação da grade processada.
- `audit_manchester_alignment.py`: auditoria multimodal.
- `audit_temporal_windows.py`: auditoria cronológica das janelas.
- `train_manchester_baseline.py`: baseline temporal com três modelos.
- `validate_manchester_lopo.py`: validação leave-one-participant-out.
- `search_feature_combinations.py`: busca de combinações multimodais.
- `requirements.txt`: dependências Python.
