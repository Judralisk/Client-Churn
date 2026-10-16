<div align="center">

# 📉 Client Churn · Análise Original

**Previsão de cancelamento de clientes com Machine Learning**

*Trabalho da disciplina de Machine Learning · Matemática · UFPR*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

</div>

---

## 🎓 Sobre este trabalho

Esta pasta guarda o projeto **exatamente como foi entregue** na disciplina de Machine Learning da UFPR.

Ele é o **ponto de partida** da versão atual do repositório, que leva o mesmo problema para um pipeline de produção
com **Databricks** e **AWS**. Toda a análise de negócio, a limpeza dos dados e a escolha de modelos nasceram aqui.

> 👉 Para a versão em produção, volte para o [README principal](../README.md).

---

## 🧩 O problema

**Churn** é a perda de clientes: a parcela que deixa de usar os produtos ou serviços de uma empresa em um período.

Saber **quem** está prestes a sair permite agir antes, com ofertas de retenção direcionadas para quem realmente vale a pena.
O objetivo do trabalho foi construir um modelo que responda:

<div align="center">

### *"Este cliente vai cancelar?"*

</div>

---

## 📊 Os dados

| | |
|---|---|
| **Base** | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle / IBM) |
| **Clientes** | 7.043 |
| **Variáveis** | 20 (perfil, serviços contratados, contrato e cobrança) |
| **Taxa de churn** | ~26,5% (base desbalanceada) |

---

## 🔬 Metodologia

```
 Dados brutos ─► Análise exploratória ─► Tratamento ─► Seleção de variáveis ─► Modelos ─► Avaliação
```

<details>
<summary><b>1 · Análise exploratória</b></summary>
<br>

- Tipagem, valores ausentes e estatísticas descritivas.
- Mapa de correlação entre as variáveis e gráficos de distribuição (violin plots e histogramas).

</details>

<details>
<summary><b>2 · Tratamento dos dados</b></summary>
<br>

- **`TotalCharges` vazio:** a investigação mostrou que só acontece em clientes com `tenure = 0`,
  ou seja, que ainda não receberam a primeira fatura. Pela regra de negócio, esses valores viraram **0**.
- Respostas equivalentes unificadas (*"No internet service"* e *"No phone service"* → *"No"*).
- Variáveis categóricas convertidas em números e remoção do `customerID` para evitar vazamento.
- Normalização min-max para eliminar problemas de escala.

</details>

<details>
<summary><b>3 · Desbalanceamento</b></summary>
<br>

- Validação **hold-out** com 70% treino e 30% validação.
- **Oversampling** da classe minoritária (`RandomOverSampler`) até 50% de churn na base de treino.

</details>

<details>
<summary><b>4 · Seleção de variáveis com Lasso</b></summary>
<br>

- Regressão Lasso com vários valores de α para ver quais coeficientes zeravam primeiro.
- Três cenários comparados: **todas as variáveis**, **6 variáveis** (α = 0,03) e **2 variáveis** (α = 0,04).
- As que mais resistiram: tipo de contrato, serviço de internet, tempo de casa, segurança online,
  fatura digital e forma de pagamento.

</details>

<details>
<summary><b>5 · Modelos</b></summary>
<br>

| Modelo | Ajuste |
|---|---|
| 🔹 KNN | Número de vizinhos via grid search |
| 🔹 Naive Bayes | Multinomial |
| 🌲 Random Forest | Número de árvores via grid search |
| ⚡ XGBoost | Taxa de aprendizado, profundidade e regularização via grid search |

Métrica principal: **recall**, porque deixar de identificar um cliente que vai sair é o erro mais caro.
Os resultados foram comparados por **matriz de confusão** em cada cenário.

</details>

---

## 📁 Arquivos

| Arquivo | O que tem |
|---|---|
| 📓 `Client_Churn_Notebook_Script.ipynb` | Notebook completo, com gráficos e saídas |
| 🐍 `Customer Churn Script.py` | Mesmo conteúdo do notebook em script Python |
| 📄 `Client_Churn_Results_Demonstration.pdf` | Relatório com todas as etapas comentadas e as conclusões |

> 💡 Os notebooks são uma análise **técnica**. Para uma leitura guiada, comece pelo **PDF**.

---

## 🚀 Do trabalho acadêmico à produção

A versão atual mantém as regras de negócio descobertas aqui e evolui a engenharia em volta delas:

| | 🎓 Análise original | ☁️ Versão Databricks + AWS |
|---|---|---|
| **Onde roda** | Google Colab | Databricks, com dados no AWS S3 |
| **Dados** | Upload manual do CSV | Camadas bronze → silver → gold em Delta |
| **Desbalanceamento** | Oversampling | Peso de classe (sem duplicar linhas) |
| **Pré-processamento** | Mapeamento manual e normalização separada | Pipeline único reaproveitado no treino e no score |
| **Métrica** | Recall | F2 (recall com controle de precisão) |
| **Modelos** | KNN, Naive Bayes, RF, XGBoost | Regressão Logística, RF, XGBoost |
| **Experimentos** | Saídas do notebook | MLflow, com histórico de cada execução |
| **Entrega** | Relatório | Modelo registrado e previsões por cliente com faixa de risco |

---

<div align="center">

Feito com ☕ e muita matriz de confusão na **UFPR**

</div>
