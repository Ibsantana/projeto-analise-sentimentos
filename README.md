# Análise de Sentimentos com Inteligência Artificial

Projeto de classificação automática de sentimentos em textos utilizando Processamento de Linguagem Natural (NLP), Machine Learning e modelos pré-treinados.

## Sobre o projeto

Este projeto tem como objetivo identificar automaticamente o sentimento presente em avaliações e textos escritos por usuários.

O sistema classifica os textos em três categorias:

- **Positivo:** demonstra satisfação ou uma experiência favorável.
- **Negativo:** demonstra insatisfação ou uma experiência desfavorável.
- **Neutro:** apresenta uma informação sem expressar claramente uma opinião positiva ou negativa.

Para realizar essa classificação, foram implementadas duas abordagens:

1. Um modelo de Machine Learning treinado com Scikit-learn.
2. Um modelo pré-treinado disponibilizado no Hugging Face.

O projeto também permite comparar os resultados dos modelos e classificar novas avaliações de forma interativa.

## Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python | Linguagem de programação |
| Pandas | Manipulação e análise dos dados |
| Scikit-learn | Treinamento e avaliação do modelo |
| TF-IDF | Vetorização dos textos |
| Logistic Regression | Classificação de sentimentos |
| Hugging Face Transformers | Utilização do modelo pré-treinado |
| PyTorch | Suporte à execução do modelo |

## Modelos implementados

### 1. Logistic Regression com TF-IDF

O primeiro classificador foi desenvolvido e treinado utilizando a biblioteca Scikit-learn.

O processo consiste em:

1. Carregar e analisar a base de dados.
2. Separar os textos e seus respectivos sentimentos.
3. Dividir os dados em treinamento e teste.
4. Transformar os textos em representações numéricas com TF-IDF.
5. Treinar o modelo Logistic Regression.
6. Realizar previsões com os dados de teste.
7. Avaliar o desempenho do classificador.

**Configuração do TF-IDF:**

- `ngram_range=(1, 2)`: considera unigramas e bigramas.
- `max_features=200`: limita o vocabulário às 200 características mais relevantes.

### 2. Modelo pré-treinado do Hugging Face

O segundo classificador utiliza o modelo:

[Horizon-Labs/multilingual-sentiment-small](https://huggingface.co/Horizon-Labs/multilingual-sentiment-small)

O modelo é carregado por meio da biblioteca Transformers e utilizado para classificar os textos sem necessidade de treiná-lo do zero neste projeto.

As classificações retornadas em inglês são convertidas para português, permitindo a comparação com as categorias da base de dados.

## Base de dados

Foi criada uma base de dados própria, armazenada em `data/sentimentos.csv`.

A base contém 90 frases elaboradas manualmente, distribuídas igualmente entre três categorias:

| Categoria | Quantidade |
|---|---:|
| Positivo | 30 |
| Negativo | 30 |
| Neutro | 30 |
| **Total** | **90** |

Os dados foram desenvolvidos para fins de estudo e experimentação.

## Etapas do desenvolvimento

O projeto foi organizado nas seguintes etapas:

1. Importação das bibliotecas.
2. Carregamento da base de dados.
3. Análise exploratória dos dados.
4. Separação das variáveis de entrada e saída.
5. Divisão dos dados em treinamento e teste.
6. Vetorização dos textos utilizando TF-IDF.
7. Criação e treinamento do Logistic Regression.
8. Realização de previsões.
9. Avaliação do primeiro modelo.
10. Carregamento do modelo pré-treinado do Hugging Face.
11. Classificação dos textos utilizando o segundo modelo.
12. Avaliação e comparação dos resultados.
13. Implementação de um classificador interativo.

## Avaliação dos modelos

Os dois modelos foram avaliados utilizando o mesmo conjunto de teste.

A divisão dos dados foi realizada com:

- 80% para treinamento.
- 20% para teste.
- `random_state=42`.
- `stratify=y`, para preservar a distribuição das classes.

### Resultados obtidos

| Modelo | Acurácia |
|---|---:|
| Logistic Regression + TF-IDF | 61,11% |
| Hugging Face | 100% |

### Matriz de confusão do Hugging Face

```text
[[6 0 0]
 [0 6 0]
 [0 0 6]]
```

O modelo classificou corretamente os 18 exemplos do conjunto de teste nessa execução.

Além da acurácia, o projeto utiliza:

- Precisão (Precision).
- Revocação (Recall).
- F1-score.
- Matriz de confusão.
- Relatório de classificação.

**Observação sobre os resultados:** a base possui apenas 90 frases elaboradas manualmente, e o conjunto de teste contém 18 exemplos. Portanto, os 100% de acurácia do Hugging Face representam somente o resultado desse experimento. Não garantem o mesmo desempenho em outros conjuntos de dados ou em aplicações reais.

## Classificação interativa

Após o treinamento e a avaliação dos modelos, o programa permite inserir novas avaliações diretamente pelo terminal.

**Exemplo de entrada:**

```text
Digite uma avaliação:
O atendimento foi excelente e resolveram tudo rapidamente.
```

**Exemplo ilustrativo de resultado:**

```text
===== RESULTADOS DA CLASSIFICAÇÃO =====

Logistic Regression: positivo
Hugging Face: positivo
Confiança do Hugging Face: valor retornado pelo modelo
```

O sistema apresenta a classificação dos dois modelos e a pontuação de confiança retornada pelo Hugging Face.

Para encerrar a execução, basta digitar:

```text
sair
```

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/Ibsantana/projeto-analise-sentimentos.git
```

### 2. Acessar a pasta do projeto

```bash
cd projeto-analise-sentimentos
```

### 3. Criar um ambiente virtual

```bash
python -m venv .venv
```

### 4. Ativar o ambiente virtual

No Windows, utilizando PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Instalar as dependências

```bash
python -m pip install -r requirements.txt
```

### 6. Executar o projeto

```bash
python main.py
```

**Importante:** na primeira execução, o modelo do Hugging Face poderá ser baixado automaticamente. É necessário ter conexão com a internet.

## Dependências

As bibliotecas utilizadas estão especificadas no arquivo `requirements.txt`:

```text
pandas==3.1.1
scikit-learn==1.9.1
transformers==5.23.0
torch==2.15.0
```

## Estrutura do projeto

```text
projeto-analise-sentimentos/
│
├── data/
│   └── sentimentos.csv
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitações

- A base de dados contém apenas 90 frases.
- Os textos foram elaborados manualmente e não representam avaliações reais.
- O conjunto de teste possui somente 18 exemplos.
- A classificação pode apresentar dificuldades com ironia, ambiguidades, gírias e erros ortográficos.
- Os resultados obtidos não devem ser generalizados para aplicações reais sem testes adicionais.

## Melhorias futuras

- Ampliar a base de dados com avaliações reais e devidamente autorizadas.
- Testar outros algoritmos de Machine Learning.
- Comparar diferentes modelos pré-treinados.
- Aprimorar o processamento dos textos.
- Avaliar os modelos utilizando bases de dados maiores.
- Desenvolver uma interface gráfica para facilitar a utilização.

## Aprendizados

Durante o desenvolvimento deste projeto, foram trabalhados conceitos importantes para a área de Inteligência Artificial:

- Manipulação e análise de dados com Pandas.
- Preparação de dados para Machine Learning.
- Processamento de Linguagem Natural.
- Vetorização de textos com TF-IDF.
- Treinamento de modelos de classificação.
- Utilização de modelos pré-treinados com Hugging Face.
- Avaliação e comparação de modelos.

## Autora

**Isabella Santana**

- [GitHub](https://github.com/Ibsantana)

---

Projeto desenvolvido para fins de estudo e aprendizado em Inteligência Artificial, Machine Learning e Processamento de Linguagem Natural.