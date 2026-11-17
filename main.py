

# ==========================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
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


# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================

dados = pd.read_csv("data/sentimentos.csv")


# ==========================================
# 3. ANÁLISE EXPLORATÓRIA DOS DADOS
# ==========================================

print("Primeiras avaliações:")
print(dados.head())

print("\nDimensão da base:")
print(dados.shape)

print("\nColunas disponíveis:")
print(dados.columns)

print("\nQuantidade de sentimentos:")
print(dados["sentimento"].value_counts())


# ==========================================
# 4. SEPARAÇÃO DAS VARIÁVEIS
# ==========================================

X = dados["texto"]
y = dados["sentimento"]


# ==========================================
# 5. DIVISÃO DOS DADOS EM TREINAMENTO E TESTE
# ==========================================

X_treino_texto, X_teste_texto, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. VETORIZAÇÃO DOS TEXTOS COM TF-IDF
# ==========================================

vetorizador = TfidfVectorizer()

# Aprende o vocabulário e transforma os textos de treinamento
X_treino = vetorizador.fit_transform(X_treino_texto)

# Transforma os textos de teste usando o vocabulário aprendido
X_teste = vetorizador.transform(X_teste_texto)


# ==========================================
# 7. CONFERÊNCIA DA DIVISÃO DOS DADOS
# ==========================================

print("\nDimensão dos dados de treinamento:", X_treino.shape)
print("Dimensão dos dados de teste:", X_teste.shape)


# ==========================================
# 8. CRIAÇÃO DO MODELO
# ==========================================

modelo = LogisticRegression()


# ==========================================
# 9. TREINAMENTO DO MODELO
# ==========================================

modelo.fit(X_treino, y_treino)

print("\nModelo treinado com sucesso!")


# ==========================================
# 10. REALIZAÇÃO DAS PREVISÕES
# ==========================================

previsoes = modelo.predict(X_teste)

print("\nPrevisões:", list(previsoes))
print("Sentimentos reais:", list(y_teste))


# ==========================================
# 11. AVALIAÇÃO DA ACURÁCIA
# ==========================================

acuracia = accuracy_score(y_teste, previsoes)

print("\nAcurácia:", acuracia)
print("Acurácia em porcentagem:", acuracia * 100, "%")


# ==========================================
# 12. MATRIZ DE CONFUSÃO
# ==========================================

matriz = confusion_matrix(
    y_teste,
    previsoes,
    labels=["negativo", "neutro", "positivo"]
)

print("\nMatriz de confusão:")
print(matriz)


# ==========================================
# 13. RELATÓRIO DE CLASSIFICAÇÃO
# ==========================================

print("\nRelatório de classificação:")

print(
    classification_report(
        y_teste,
        previsoes,
        zero_division=0
    )
)


# ==========================================
# 14. CLASSIFICADOR INTERATIVO
# ==========================================

while True:

    texto_novo = input("\nDigite uma avaliação (ou 'sair' para encerrar): ")

    # Verificar se o usuário deseja encerrar
    if texto_novo.strip().lower() == "sair":
        print("Programa encerrado!")
        break

    # Verificar se o texto está vazio
    if texto_novo.strip() == "":
        print("Digite uma avaliação válida!")
        continue

    # Transformar o texto usando o TF-IDF já treinado
    texto_novo_vetorizado = vetorizador.transform([texto_novo])

    # Prever o sentimento da avaliação
    previsao_nova = modelo.predict(texto_novo_vetorizado)

    # Exibir o resultado
    print("Sentimento identificado:", previsao_nova[0])


