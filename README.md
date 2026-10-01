# Previsão de Sílica no Concentrado da Flotação de Minério de Ferro

Previsão, com 1 hora de antecedência, do `% Silica Concentrate` (nível de impureza no concentrado da flotação) a partir de dados de sensores do processo, com foco em **qualidade dos dados, engenharia de features sem vazamento e comparação honesta com baselines**.

> **Resumo:** depois de limpar cerca de 38% dos dados horários (leituras interpoladas e sensores "travados") e construir features de lag/janela móvel usando apenas informação passada, um modelo linear regularizado (Ridge) superou o XGBoost ajustado e o Random Forest em RMSE. O baseline ingênuo ("nada muda") se mostrou um concorrente muito forte, empatando com o Ridge no MAE.

---

## Problema

Em uma planta de flotação, o teor de sílica no concentrado é medido pelo laboratório apenas uma vez por hora. Antecipar esse valor com 1 hora de antecedência permitiria ao operador ajustar reagentes (amido, amina) e parâmetros das colunas antes que a qualidade saia da especificação.

**Tarefa:** prever o `% Silica Concentrate` na hora `t` usando apenas informações disponíveis **antes** da hora `t`.

## Dataset

[Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process) (Kaggle): dados contínuos de uma planta de flotação de **10/03/2017 a 09/09/2017**, amostrados a cada 20 segundos (737.453 linhas brutas), com `% Iron/Silica Feed` e `% Iron/Silica Concentrate` amostrados de hora em hora no laboratório.

O arquivo usa vírgula como separador decimal, por isso a leitura é feita com `decimal=','`.

## Abordagem

### 1. Limpeza dos dados (funções reutilizáveis)

Dois problemas de qualidade independentes são tratados por funções que podem ser reaplicadas a novos dados:

| Problema | Detecção | Ação |
|---|---|---|
| **Interpolação dentro da hora** | Os alvos de laboratório (`conc_silica`, `conc_iron`) deveriam ser constantes dentro de cada hora. Mais de um valor único na mesma hora indica suavização artificial. | `remove_midhour_interpolation()` descarta a hora inteira (4.097 → 3.777 horas). |
| **Leituras de alimentação "travadas" (forward-fill)** | `feed_iron` / `feed_silica` são amostradas pelo laboratório aproximadamente a cada 9 horas, então platôs curtos são legítimos. Sequências de valores idênticos por até 736 horas (30 dias) não são. | `flag_stale_runs()` marca sequências maiores que 48 horas, que são descartadas. |

Após a agregação para resolução horária (média para os sensores de 20 segundos, `first()` para os valores de laboratório já constantes), restam **2.539 horas**.

### 2. Engenharia de features (sem vazamento)

Todas as features de lag e janela móvel são construídas a partir de `shift(1)`, de modo que uma linha nunca enxerga informação da hora que está sendo prevista.

- **Histórico do alvo:** lags 1, 2 e 3; média móvel (3h, 6h, 12h); desvio padrão móvel (6h)
- **Variáveis de processo:** lag 1 e média móvel (3h, 6h) para `amina_flow`, `starch_flow`, `level_col6`, `level_col7`, `air_col1`, `air_col6`, `pulp_ph`, `feed_iron` e `feed_silica`

**Verificação de contiguidade temporal:** a limpeza deixa lacunas no índice horário, e uma janela de lag/média móvel que atravessa uma lacuna mistura, sem avisar, horas não adjacentes. Cada linha é validada contra o índice horário para confirmar que sua janela completa de 12 horas é realmente consecutiva; as que falham são descartadas.

Dataset supervisionado final: **1.940 linhas × 34 features**.

### 3. Divisão temporal (sem embaralhar)

| Conjunto | Linhas | Período |
|---|---|---|
| Treino | 1.358 | 10/03/2017 → 18/07/2017 |
| Validação | 291 | 18/07/2017 → 23/08/2017 |
| Teste | 291 | 23/08/2017 → 09/09/2017 |

O modelo é sempre avaliado em dados estritamente posteriores aos usados no treino e no ajuste, como ocorreria em produção. Os hiperparâmetros do XGBoost foram ajustados apenas na validação; o conjunto de teste nunca foi usado para seleção.

### 4. Modelos comparados

1. **Persistência ingênua** (baseline): prever a hora `t` como o valor real da hora `t-1`
2. **Regressão Ridge** (`alpha=5.0`)
3. **Random Forest** (500 árvores, `max_depth=6`, `min_samples_leaf=5`)
4. **XGBoost**, com grid search na validação (melhor: `max_depth=2`, `learning_rate=0.02`, `n_estimators=200`, `subsample=1.0`)

## Resultados (conjunto de teste)

| Modelo | MAE | RMSE |
|---|---|---|
| **Ridge** | 0,5216 | **0,7444** |
| XGBoost (ajustado) | 0,5499 | 0,7652 |
| Random Forest | 0,5850 | 0,7866 |
| Persistência ingênua | **0,5200** | 0,8001 |

### O que os resultados mostram

- **O baseline ingênuo é realmente forte.** O processo muda devagar, então o `% Silica Concentrate` tem alta autocorrelação e "prever que nada muda" já funciona bem. Sem esse baseline, um RMSE de 0,77 no XGBoost pareceria ótimo; com ele, o ganho real é modesto.
- **O Ridge reduz o RMSE em cerca de 7% em relação à persistência** (0,744 vs 0,800), mas o **MAE é praticamente empatado** (0,522 vs 0,520). O ganho vem da redução de erros grandes, que o RMSE penaliza mais, e não de um erro típico menor.
- **O modelo mais simples venceu.** Com cerca de 1.900 linhas, o modelo linear superou os de árvore. Isso não prova que modelos de árvore não agregam valor; com um histórico maior da planta, o ranking pode mudar.
- **A autocorrelação domina o sinal.** Os maiores coeficientes do Ridge estão no histórico recente do próprio alvo (`conc_silica_lag1`, `conc_silica_roll3mean`, `conc_silica_roll6mean`), seguidos por contribuições menores de features de `pulp_ph`, `feed_iron` e `feed_silica`.

## Limitações

- A etapa de limpeza remove cerca de 40% dos dados horários. Um limiar de sensor "travado" mais brando pode recuperar dados úteis sem reintroduzir leituras ruins.
- Apenas o **horizonte de 1 hora** foi testado.
- O conjunto de teste é pequeno (291 horas, ~17 dias), então diferenças entre modelos na faixa de 0,02 de RMSE devem ser lidas com cautela.
- A vantagem do Ridge sobre XGBoost / Random Forest é modesta e deve ser retestada com mais dados.

## Próximos passos

- Previsão multi-step (2 a 6 horas à frente), mais útil na operação
- Encapsular limpeza e engenharia de features em um `sklearn.Pipeline` ou em um par `prep.py` / `train.py`
- Rastreamento de experimentos com **MLflow**
- Servir o modelo escolhido por meio de um endpoint **FastAPI**

## Como executar

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

1. Baixe o dataset no [Kaggle](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process).
2. No notebook, ajuste `DATA_PATH` (seção 2) para o local do arquivo `MiningProcess_Flotation_Plant_Database.csv`.
3. Execute `silica_forecast_combined.ipynb` do início ao fim.

## Conteúdo do repositório

```
silica_forecast_combined.ipynb   # limpeza, features, divisão, modelos, avaliação
```

## Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib

## Autor

**Antenor**, analista de dados. [LinkedIn](#) · [GitHub](#)
