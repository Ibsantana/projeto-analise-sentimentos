
# ==========================================
# 1. IMPORTAÇÕES
# ==========================================

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from transformers import pipeline


# ==========================================
# 2. CONFIGURAÇÕES
# ==========================================

ROTULOS = ["negativo", "neutro", "positivo"]

TRADUCAO_SENTIMENTOS = {
    "positive": "positivo",
    "negative": "negativo",
    "neutral": "neutro"
}


# ==========================================
# 3. CARREGAMENTO E ANÁLISE DOS DADOS
# ==========================================

dados = pd.read_csv("data/sentimentos.csv")

print("Primeiras avaliações:")
print(dados.head())

print("\nDimensão da base:")
print(dados.shape)

print("\nColunas disponíveis:")
print(dados.columns)

print("\nQuantidade de sentimentos:")
print(dados["sentimento"].value_counts())


# ==========================================
# 4. PREPARAÇÃO DOS DADOS
# ==========================================

X = dados["texto"]
y = dados["sentimento"]

X_treino_texto, X_teste_texto, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nQuantidade de avaliações negativas no treinamento:")

for texto, sentimento in zip(X_treino_texto, y_treino):
    if sentimento == "negativo":
        print("-", texto)


# ==========================================
# 5. VETORIZAÇÃO COM TF-IDF
# ==========================================

vetorizador = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=200
)

X_treino = vetorizador.fit_transform(X_treino_texto)
X_teste = vetorizador.transform(X_teste_texto)

print("\nQuantidade de características:")
print(len(vetorizador.get_feature_names_out()))

print("\nDimensão dos dados de treinamento:", X_treino.shape)
print("Dimensão dos dados de teste:", X_teste.shape)


# ==========================================
# 6. TREINAMENTO DO LOGISTIC REGRESSION
# ==========================================

modelo = LogisticRegression()

modelo.fit(X_treino, y_treino)

print("\nModelo treinado com sucesso!")

previsoes = modelo.predict(X_teste)

print("\nPrevisões:", list(previsoes))
print("Sentimentos reais:", list(y_teste))

print("\nAvaliações classificadas incorretamente:")

for texto, real, previsto in zip(
    X_teste_texto,
    y_teste,
    previsoes
):
    if real != previsto:
        print("\nTexto:", texto)
        print("Sentimento real:", real)
        print("Sentimento previsto:", previsto)


# ==========================================
# 7. AVALIAÇÃO DO LOGISTIC REGRESSION
# ==========================================

acuracia = accuracy_score(y_teste, previsoes)

print("\n===== RESULTADO DO LOGISTIC REGRESSION =====")

print("Acurácia:", round(acuracia * 100, 2), "%")

print("\nMatriz de confusão:")
print(confusion_matrix(
    y_teste,
    previsoes,
    labels=ROTULOS
))

print("\nRelatório de classificação:")
print(classification_report(
    y_teste,
    previsoes,
    labels=ROTULOS,
    zero_division=0
))


# ==========================================
# 8. CARREGAMENTO DO HUGGING FACE
# ==========================================

print("\nCarregando modelo pré-treinado do Hugging Face...")

classificador_hf = pipeline(
    "text-classification",
    model="Horizon-Labs/multilingual-sentiment-small"
)

print("Modelo do Hugging Face carregado com sucesso!")


# ==========================================
# 9. PREVISÕES DO HUGGING FACE
# ==========================================

previsoes_hf = []

for texto in X_teste_texto:

    resultado = classificador_hf(texto)

    sentimento_ingles = resultado[0]["label"].lower()

    sentimento_portugues = TRADUCAO_SENTIMENTOS.get(
        sentimento_ingles,
        sentimento_ingles
    )

    previsoes_hf.append(sentimento_portugues)


# ==========================================
# 10. AVALIAÇÃO DO HUGGING FACE
# ==========================================

acuracia_hf = accuracy_score(y_teste, previsoes_hf)

print("\n===== RESULTADO DO HUGGING FACE =====")

print("Acurácia:", round(acuracia_hf * 100, 2), "%")

print("\nMatriz de confusão:")
print(confusion_matrix(
    y_teste,
    previsoes_hf,
    labels=ROTULOS
))

print("\nRelatório de classificação:")
print(classification_report(
    y_teste,
    previsoes_hf,
    labels=ROTULOS,
    zero_division=0
))


# ==========================================
# 11. COMPARAÇÃO DOS MODELOS
# ==========================================

print("\n===== COMPARAÇÃO DOS MODELOS =====")

print(
    "Logistic Regression:",
    round(acuracia * 100, 2),
    "%"
)

print(
    "Hugging Face:",
    round(acuracia_hf * 100, 2),
    "%"
)


# ==========================================
# 12. CLASSIFICAÇÃO INTERATIVA
# ==========================================

print("\n===== CLASSIFICADOR DE SENTIMENTOS =====")

while True:

    texto_novo = input(
        "\nDigite uma avaliação (ou 'sair' para encerrar): "
    )

    if texto_novo.strip().lower() == "sair":
        print("Programa encerrado!")
        break

    if not texto_novo.strip():
        print("Digite uma avaliação válida!")
        continue

    # Logistic Regression
    texto_vetorizado = vetorizador.transform([texto_novo])
    previsao_lr = modelo.predict(texto_vetorizado)

    # Hugging Face
    resultado_hf = classificador_hf(texto_novo)

    sentimento_ingles = resultado_hf[0]["label"].lower()

    sentimento_portugues = TRADUCAO_SENTIMENTOS.get(
        sentimento_ingles,
        sentimento_ingles
    )

    confianca = resultado_hf[0]["score"]

    print("\n===== RESULTADOS DA CLASSIFICAÇÃO =====")

    print("Logistic Regression:", previsao_lr[0])
    print("Hugging Face:", sentimento_portugues)

    print(
        "Confiança do Hugging Face:",
        round(confianca * 100, 2),
        "%"
    )
