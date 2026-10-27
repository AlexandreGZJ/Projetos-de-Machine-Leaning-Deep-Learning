# Shark ML — Classificação com MLP

Código organizado a partir das células do notebook original.

> Observação: as células foram mantidas na ordem original. Para executar o projeto, mantenha também o diretório `sharks_DB` com as imagens organizadas em subpastas por espécie.

---

### Célula 1

```python
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
```

---

### Célula 2

```python
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
```

---

### Célula 3

```python
pasta_dataset = Path("sharks_DB")
```

---

### Célula 4

```python
print(pasta_dataset)
```

---

### Célula 5

```python
especies = [pasta for pasta in pasta_dataset.iterdir() if pasta.is_dir()]
```

---

### Célula 6

```python
print("Quantidade de espécies:", len(especies))

for especie in sorted(especies):
    print(especie.name)
```

---

### Célula 7

```python
extensoes_validas = {".jpg", ".jpeg", ".png", ".webp"}

for especie in sorted(pasta_dataset.iterdir()):
    if especie.is_dir():
        imagens = [
            arquivo for arquivo in especie.iterdir()
            if arquivo.suffix.lower() in extensoes_validas
        ]

        print(f"{especie.name}: {len(imagens)} imagens")
```

---

### Célula 8

```python
quantidades = []

for especie in sorted(pasta_dataset.iterdir()):
    if especie.is_dir():
        imagens = [
            arquivo for arquivo in especie.iterdir()
            if arquivo.suffix.lower() in extensoes_validas
        ]

        quantidades.append(len(imagens))

print("Total de imagens:", sum(quantidades))
print("Total de espécies:", len(quantidades))
print("Média por espécie:", round(sum(quantidades) / len(quantidades), 1))
```

---

### TESTE COM IMAGENS PARA VERIFICAR DEFORMAÇÃO

```python
# TESTE COM IMAGENS PARA VERIFICAR DEFORMAÇÃO
```

---

### Seleciona a primeira espécie e uma imagem dessa pasta

```python
# Seleciona a primeira espécie e uma imagem dessa pasta
pasta_especie = next(
    pasta for pasta in sorted(pasta_dataset.iterdir())
    if pasta.is_dir()
)

arquivo_imagem = next(
    arquivo for arquivo in pasta_especie.iterdir()
    if arquivo.suffix.lower() in extensoes_validas
)

# Abre a imagem
imagem = Image.open(arquivo_imagem)

# Mostra informações
print("Espécie:", pasta_especie.name)
print("Arquivo:", arquivo_imagem.name)
print("Resolução original:", imagem.size)
print("Modo de cor:", imagem.mode)

# Exibe a imagem
plt.imshow(imagem)
plt.axis("off")
plt.show()
```

---

### Redimensiona para 224 x 224 pixels

```python
# Redimensiona para 224 x 224 pixels
imagem_redimensionada = imagem.resize((224, 224))

# Mostra a nova resolução
print("Resolução original:", imagem.size)
print("Nova resolução:", imagem_redimensionada.size)

# Exibe a imagem redimensionada
plt.imshow(imagem_redimensionada)
plt.axis("off")
plt.show()
```

---

### COLOCANDO BORDAS PRETAS EM TODAS AS IMAGENS PARA UNIFORMIZAR LEITURA

```python
# COLOCANDO BORDAS PRETAS EM TODAS AS IMAGENS PARA UNIFORMIZAR LEITURA
```

---

### Célula 13

```python
imagem_proporcional = imagem.copy()
imagem_proporcional.thumbnail((224, 224))

# Cria um fundo quadrado preto
imagem_final = Image.new("RGB", (224, 224), (0, 0, 0))

# Calcula a posição para centralizar a imagem
x = (224 - imagem_proporcional.width) // 2
y = (224 - imagem_proporcional.height) // 2

# Cola a imagem sobre o fundo
imagem_final.paste(imagem_proporcional, (x, y))

# Exibe o resultado
print("Tamanho da imagem:", imagem_final.size)

plt.imshow(imagem_final)
plt.axis("off")
plt.show()
```

---

### Célula 14

```python
X = []
y = []

# Tamanho padronizado das imagens
TAMANHO = (224, 224)

# Extensões de imagem aceitas
extensoes_validas = {".jpg", ".jpeg", ".png", ".webp"}

# Contadores para acompanhar o processamento
total_processadas = 0
total_ignoradas = 0

# Percorrer as pastas das espécies em ordem alfabética
for indice, pasta_especie in enumerate(sorted(pasta_dataset.iterdir())):

    # Ignorar itens que não sejam pastas
    if not pasta_especie.is_dir():
        continue

    # Percorrer os arquivos da espécie
    for arquivo in pasta_especie.iterdir():

        # Ignorar arquivos com extensões não aceitas
        if arquivo.suffix.lower() not in extensoes_validas:
            continue

        try:
            with Image.open(arquivo) as imagem:

                # Garantir que a imagem tenha um canal de transparência
                imagem = imagem.convert("RGBA")

                # Redimensionar sem deformar
                imagem.thumbnail(TAMANHO)

                # Criar um fundo preto opaco
                imagem_final = Image.new(
                    "RGBA",
                    TAMANHO,
                    (0, 0, 0, 255)
                )

                # Calcular a posição para centralizar
                x = (224 - imagem.width) // 2
                y_pos = (224 - imagem.height) // 2

                # Colocar a imagem sobre o fundo, respeitando a transparência
                imagem_final.paste(imagem, (x, y_pos), imagem)

                # Converter para RGB, removendo o canal alfa
                imagem_final = imagem_final.convert("RGB")

                # Transformar os pixels em números entre 0 e 1
                pixels = np.asarray(
                    imagem_final,
                    dtype=np.float32
                ) / 255.0

                # Guardar a imagem e o rótulo da espécie
                X.append(pixels)
                y.append(indice)

                total_processadas += 1

        except (OSError, ValueError) as erro:
            print(f"Imagem ignorada: {arquivo} | Motivo: {erro}")
            total_ignoradas += 1

# Converter as listas em arrays NumPy
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Exibir o resumo
print("\n--- Resultado do processamento ---")
print("Imagens processadas:", total_processadas)
print("Imagens ignoradas:", total_ignoradas)
print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Quantidade de rótulos:", len(np.unique(y)))
```

---

### Célula 15

```python
classes = [
    pasta.name
    for pasta in sorted(pasta_dataset.iterdir())
    if pasta.is_dir()
]

for indice, nome in enumerate(classes):
    print(f"Rótulo {indice}: {nome}")
```

---

### VISUALIZAR IMAGENS PROCESSADAS

```python
#VISUALIZAR IMAGENS PROCESSADAS
```

---

### Quantidade de imagens que queremos visualizar

```python
# Quantidade de imagens que queremos visualizar
quantidade = 5

plt.figure(figsize=(15, 4))

for i in range(quantidade):
    plt.subplot(1, quantidade, i + 1)

    # Exibe a imagem processada
    plt.imshow(X[i])

    # Converte o rótulo numérico para o nome da espécie
    plt.title(classes[y[i]])

    plt.axis("off")

plt.tight_layout()
plt.show()
```

---

### Célula 18

```python
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Imagens de treinamento:", len(X_treino))
print("Imagens de teste:", len(X_teste))

print("Rótulos de treinamento:", len(y_treino))
print("Rótulos de teste:", len(y_teste))
```

---

### Célula 19

```python
from collections import Counter

print("Distribuição no treinamento:")
for indice, quantidade in sorted(Counter(y_treino).items()):
    print(f"{classes[indice]}: {quantidade}")

print("\nDistribuição no teste:")
for indice, quantidade in sorted(Counter(y_teste).items()):
    print(f"{classes[indice]}: {quantidade}")
```

---

### COMEÇA A USAR O MLP / TRANSFORMANDO IMAGEM EM VALORES

```python
#COMEÇA A USAR O MLP / TRANSFORMANDO IMAGEM EM VALORES
```

---

### Selecionar a primeira imagem

```python
# Selecionar a primeira imagem
imagem_exemplo = X[0]

# Transformar a imagem em um vetor unidimensional
vetor_exemplo = imagem_exemplo.flatten()

print("Formato original:", imagem_exemplo.shape)
print("Formato do vetor:", vetor_exemplo.shape)
print("Quantidade de características:", len(vetor_exemplo))
```

---

### TRANSOFRMA TD EM X EM VETOR

```python
#TRANSOFRMA TD EM X EM VETOR
```

---

### Célula 23

```python
X_mlp = X.reshape(X.shape[0], -1)

print("Formato dos dados:", X_mlp.shape)
print("Formato dos rótulos:", y.shape)
```

---

### Célula 24

```python
X_treino_mlp, X_teste_mlp, y_treino, y_teste = train_test_split(
    X_mlp,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Treinamento:", X_treino_mlp.shape)
print("Teste:", X_teste_mlp.shape)
```

---

### MLP INICIO

```python
#MLP INICIO
```

---

### Célula 26

```python
modelo_mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    max_iter=100,
    random_state=42,
    verbose=True
)

print("Modelo MLP configurado.")
print("Camadas ocultas:", modelo_mlp.hidden_layer_sizes)
print("Máximo de iterações:", modelo_mlp.max_iter)
```

---

### Célula 27

```python
inicio = time.perf_counter()

modelo_mlp.fit(X_treino_mlp, y_treino)

time.sleep(1.5)
fim = time.perf_counter()

tempo_exec = fim - inicio
```

---

### Célula 28

```python
from sklearn.metrics import accuracy_score

y_pred = modelo_mlp.predict(X_teste_mlp)

acuracia = accuracy_score(y_teste, y_pred)

print(f"Acurácia no teste: {acuracia:.2%}")
print(f"Tempo de execução: {tempo_exec:.2f} segundos!")
```

---

### Célula 29

```python
from sklearn.metrics import accuracy_score

y_pred_treino = modelo_mlp.predict(X_treino_mlp)
y_pred_teste = modelo_mlp.predict(X_teste_mlp)

print(
    f"Acurácia no treinamento: "
    f"{accuracy_score(y_treino, y_pred_treino):.2%}"
)

print(
    f"Acurácia no teste: "
    f"{accuracy_score(y_teste, y_pred_teste):.2%} "   
)

print(f"Tempo de execução: {tempo_exec:.2f} segundos!")

print(f"Loss final: {modelo_mlp.loss_:.4f}")
print(f"Iterações realizadas: {modelo_mlp.n_iter_}")
```
