# Detecção de Intrusão em Fluxos de Dados: Avaliação Comportamental e Otimização

[![Conferência](https://img.shields.io/badge/SBCSeg-Artigo_Aceito-blue)](https://sbc.org.br/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![CapyMOA](https://img.shields.io/badge/Framework-CapyMOA-orange)](https://capymoa.org/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-green)](https://optuna.org/)

Este repositório contém o código-fonte, os experimentos e as implementações desenvolvidas para o artigo submetido/aceito no **SBCSeg (Simpósio Brasileiro de Segurança da Informação e de Sistemas Computacionais)**. 

O projeto propõe um framework robusto para a avaliação de modelos de Machine Learning aplicados à detecção de intrusão em cenários de **fluxo contínuo de dados (Data Streams)**, enfrentando os desafios de *Concept Drift* gerados por ondas de ataques cibernéticos.

## Principais Contribuições e Funcionalidades

1. **Pipeline de Detecção Híbrida**: Suporte nativo para modelos de **Classificação Supervisionada** e **Detecção de Anomalias Não-Supervisionada/Semi-Supervisionada** processando instâncias de rede em tempo real.
2. **Otimização Contínua**: Integração com o `Optuna` para busca de hiperparâmetros (incluindo *thresholding* dinâmico) diretamente sobre arquiteturas de fluxo contínuo.
3. **Métricas Comportamentais (Inovação)**: Além das métricas tradicionais (F1-Score, Precision, Recall, MCC, FPR, TPR), o framework implementa avaliação vetorial de comportamento sob ataques:
   - **Métrica de Passagem:** Mede a queda ou ganho de performance (F1-Score) do modelo durante a ocorrência da onda de ataque.
   - **Métrica de Recuperação:** Mede a resiliência e a capacidade de adaptação (aprendizado contínuo) do algoritmo $X$ amostras após o término do ataque.
4. **Visualização Avançada**: Geração automática de relatórios cumulativos e plotagem de métricas evolutivas correlacionadas graficamente com as regiões de ataque do dataset.

## Estrutura do Repositório

A arquitetura do projeto foi desenvolvida com foco em modularidade e reutilização acadêmica:

```text
📦 anomaly-and-classification-data-stream
 ┣ 📂 src/
 ┃ ┣ 📂 Anomaly/               # Módulo de Detecção de Anomalias
 ┃ ┃ ┣ 📜 Models.py            # Definição de Modelos (AE, AIF, HST)
 ┃ ┃ ┣ 📜 Optimizer.py         # Otimização via Optuna (Threshold Dinâmico/Params)
 ┃ ┃ ┗ 📜 Pipeline.py          # Runner de avaliação Prequencial
 ┃ ┣ 📂 Classification/        # Módulo de Classificação
 ┃ ┃ ┣ 📜 Models.py            # Definição de Modelos (LB, HAT, ARF, HT)
 ┃ ┃ ┣ 📜 Optimizer.py         # Otimização de Hiperparâmetros via Optuna
 ┃ ┃ ┗ 📜 Pipeline.py          # Runner de avaliação Prequencial
 ┃ ┣ 📂 Data/
 ┃ ┃ ┗ 📜 Processor.py         # Ingestão, Normalização e Pré-processamento 
 ┃ ┗ 📂 Results/
 ┃   ┣ 📜 Metrics.py           # Cálculo de Métricas Sklearn e Comportamentais
 ┃   ┗ 📜 Plots.py             # Gráficos Evolutivos e de Score de Anomalia
 ┣ 📜 AnomalyDetection.ipynb   # Notebook de Execução (Anomalias)
 ┣ 📜 Classification.ipynb     # Notebook de Execução (Classificação)
 ┣ 📜 Database.ipynb           # Notebook de exploração de base de dados
 ┣ 📜 requirements.txt         # Dependências do projeto
 ┗ 📜 README.md                # Documentação do Repositório
```

## Modelos Avaliados

1. Detecção de Anomalias:
    - Autoencoder (AE) - Treinamento restrito ao fluxo de tráfego normal.
    - Adaptive Isolation Forest (AIF)
    - Half-Space Trees (HST)

2. Classificação:
    - Leveraging Bagging (LB)
    - Hoeffding Adaptive Tree (HAT)
    - Adaptive Random Forest (ARF)
    - Hoeffding Tree (HT)


## Instalação e Configuração

Pré-requisitos: Python 3.9+ e Java (JRE/JDK) instalado em sua máquina, necessário para a execução do backend MOA.

1. Clone este repositório:

```code
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio
```

2. Crie um ambiente virtual (recomendado):

```code
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```code
pip install -r requirements.txt
```


## Como Utilizar

O pipeline foi projetado para ser executado de forma simples através dos Jupyter Notebooks disponíveis na raiz do projeto:

- **AnomalyDetection.ipynb:** Carrega o dataset, processa o fluxo instanciando o DataStreamProcessor, e executa a otimização de parâmetros com a exibição do Relatório Comportamental.
- **Classification.ipynb:** Segue a mesma lógica estrutural, mas aplica os algoritmos supervisionados em regime prequencial (Test-then-Train).


Exemplo de Instanciação do Otimizador (Anomalias):

```code
    from src.Anomaly.Optimizer import AnomalyOptunaOptimizer
    
    optimizer = AnomalyOptunaOptimizer(
        stream=stream,
        n_trials=10,
        discretization_threshold='params', # ou 'dinamic', ou valor float (ex: 0.8)
        target_class='macro',              # Métricas globais
        target_class_pass=0,               # Alvo para métricas de passagem/recuperação
        target_names=targets
    )
    
    melhor_modelo = optimizer.optimize('HST', warmup_instances=1000)
```

## Relatórios e Saídas
Ao final da execução, o framework gera de forma automática:

- Tabela de Otimização Optuna: Melhores hiperparâmetros encontrados.
- Relatório Cumulativo: Contendo F1-Score, Precision, Recall, MCC, FPR e TPR gerais.
- Relatório Comportamental por Ataque
- Gráficos de Linha Prequenciais: Evolução temporal das métricas do modelo e limiares dinâmicos cruzados com regiões sombreadas indicando as ondas de ataque.






















