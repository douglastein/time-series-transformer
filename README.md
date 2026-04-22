# Previsão de Séries Temporais com Transformer em PyTorch

Este projeto implementa uma arquitetura de Rede Neural baseada em **Transformer** (Attention Is All You Need) construída do zero em PyTorch para a tarefa de **Forecasting de Séries Temporais**. 

Diferente de modelos tradicionais (como ARIMA) ou redes recorrentes, este modelo utiliza o mecanismo de *Multi-Head Attention* para identificar padrões complexos e dependências de longo prazo em dados sequenciais.

## Como o Modelo Funciona

O projeto resolve o problema de previsão transformando uma série contínua em um problema de aprendizado supervisionado usando a arquitetura Transformer através de janelas deslizantes:

1. O modelo observa os últimos `N` dias (ex: 50 dias).
2. Ele tenta prever o dia `N + 1`.
3. Para prever passos futuros, o modelo assume uma postura autorregressiva, consumindo a própria previsão anterior para dar o próximo passo.

### A Matemática por Trás
Como o Transformer processa toda a janela de tempo de uma vez e não possui recorrência embutida, implementamos uma camada de **Positional Encoding** baseada em funções de seno e cosseno para injetar a noção de tempo e ordem matemática nos dados.

## Tecnologias Utilizadas

* **Linguagem:** 3.14.3
* **Deep Learning:** PyTorch (`torch`, `torch.nn`)
* **Pré-processamento:** Scikit-learn (`MinMaxScaler`)
* **Métricas:** Scikit-learn (`MSE`, `RMSE`, `MAE`)
* **Visualização:** Matplotlib, Numpy

## 📁 Estrutura do Projeto

```text
meu_projeto/
│
├── src/
│   ├── data_prep.py     # Lógica de Janela Deslizante e DataLoaders
│   └── model.py         # Classes PositionalEncoding e TimeSeriesTransformer
│
├── main.py              # Script principal de treinamento e avaliação
├── README.md            # Documentação do projeto
└── requirements.txt     # Dependências