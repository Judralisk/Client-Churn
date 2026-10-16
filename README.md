# Client Churn · Databricks + AWS

Modelo que prevê quais clientes de uma operadora de telecom vão cancelar o serviço (churn),
rodando no **Databricks** com dados e tabelas armazenados no **AWS S3**.

Evolução do trabalho de Machine Learning da UFPR (análise original em [`analise_original/`](analise_original/)),
agora como pipeline de produção: ingestão em camadas, rastreio de experimentos, registro de modelo e score em lote.

## Arquitetura

```
 S3 raw/                 Databricks (Unity Catalog, tabelas Delta em S3 uc/)
 telco_customer_churn.csv
        │
        ▼
  [01 bronze] ──► bronze_clientes   CSV como veio + metadados de ingestão
        │
        ▼
  [02 silver] ──► silver_clientes   tipagem, limpeza, checagens de qualidade
        │
        ▼
  [03 treino] ──► MLflow            3 modelos, grid search, métricas e matriz de confusão
        │          churn_model      melhor modelo no Unity Catalog, alias @champion
        ▼
  [04 score]  ──► gold_previsoes_churn   probabilidade e faixa de risco por cliente ativo
```

Tudo roda como um único job (`churn-pipeline`) definido em código com Databricks Asset Bundles.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `src/churn/` | Código do modelo: features, pré-processamento, treino e avaliação |
| `src/notebooks/` | Notebooks de cada etapa do pipeline |
| `resources/churn_job.yml` | Definição do job no Databricks |
| `databricks.yml` | Configuração do bundle (workspace, catálogo, bucket) |
| `infra/aws_setup.sh` | Cria o bucket S3 e sobe a base |
| `data/` | Base [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7.043 clientes) |
| `tests/` | Teste local do treino, sem precisar do Databricks |

## Modelo

- **Candidatos:** Regressão Logística, Random Forest e XGBoost, com grid search e validação cruzada de 5 folds.
- **Métrica de escolha:** F2, que pesa mais o recall: deixar passar um cliente que vai sair custa mais
  do que oferecer retenção a quem ficaria.
- **Desbalanceamento** (26,5% de churn): tratado com peso de classe, sem duplicar linhas.
- **Validação:** hold-out estratificado de 30%, como na análise original.

Resultado no hold-out (teste local):

| Modelo | Recall | Precisão | F2 | ROC AUC |
|---|---|---|---|---|
| XGBoost | 0,82 | 0,52 | 0,74 | 0,84 |
| Random Forest | 0,81 | 0,51 | 0,72 | 0,84 |
| Regressão Logística | 0,80 | 0,51 | 0,72 | 0,84 |

O modelo encontra cerca de 8 em cada 10 clientes que vão cancelar.

## Como rodar

### 1. Ferramentas
- [AWS CLI](https://aws.amazon.com/cli/) configurado (`aws configure`)
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) logado: `databricks auth login --host https://SEU-WORKSPACE.cloud.databricks.com`

### 2. Bucket S3 e base de dados
```bash
bash infra/aws_setup.sh NOME-DO-BUCKET
```

### 3. Ligar o bucket ao Unity Catalog
No Databricks: **Catalog → External data → External locations → Create external location → AWS Quickstart**.
Informe `s3://NOME-DO-BUCKET` e siga o CloudFormation. Ele cria a role IAM e a credencial automaticamente.

### 4. Configurar o bundle
Em `databricks.yml`, preencha `host`, `catalog` (nome do seu catálogo no Catalog Explorer) e `bucket`.

### 5. Deploy e execução
```bash
databricks bundle deploy
databricks bundle run churn_pipeline
```

O job aparece em **Workflows**, os experimentos em **Experiments** e o modelo em **Catalog → SEU_CATALOGO → churn → Models**.

### Teste local
```bash
pip install -r requirements.txt
pytest -s tests
```
