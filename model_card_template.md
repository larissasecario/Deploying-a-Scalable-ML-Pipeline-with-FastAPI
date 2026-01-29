# Model Card

For additional information see the Model Card paper: [https://arxiv.org/pdf/1810.03993.pdf](https://arxiv.org/pdf/1810.03993.pdf)

---

## Model Details

Este é um modelo de Machine Learning de **classificação binária** criado para prever se a renda anual de uma pessoa é **maior que 50 mil dólares (>50K)** ou **menor ou igual a 50 mil dólares (≤50K)**.

O modelo foi treinado usando dados demográficos e profissionais do conjunto **Census Income (Adult Dataset)**.
Variáveis como educação, tipo de trabalho, ocupação, estado civil, sexo e país de origem são usadas como entrada para o modelo.

O pipeline inclui:

- Processamento de dados (One-Hot Encoding para variáveis categóricas)
- Treinamento de modelo supervisionado
- Avaliação de desempenho geral e por grupos (fatias de dados)

---

## Intended Use

Este modelo foi desenvolvido **para fins educacionais** e para demonstrar um pipeline completo de Machine Learning.

Ele pode ser usado para:

- Estudar classificação binária
- Avaliar métricas como precisão, recall e F1-score
- Analisar como o desempenho do modelo varia entre diferentes grupos da população

**Este modelo não deve ser utilizado para decisões reais** que afetem pessoas, como contratação, crédito, benefícios ou qualquer outro processo sensível.

---

## Training Data

O modelo foi treinado com o conjunto de dados **Census Income (Adult Dataset)**.

Cada registro representa uma pessoa e inclui informações como:

- Idade
- Tipo de trabalho (_workclass_)
- Nível de educação
- Estado civil
- Ocupação
- Relacionamento
- Raça
- Sexo
- País de origem

A variável alvo é:

- **salary** — indica se a renda anual é maior que 50K ou não.

Os dados foram divididos em:

- **80% para treino**
- **20% para teste**

---

## Evaluation Data

A avaliação foi feita usando o conjunto de **teste (20%)**, que não foi usado no treinamento do modelo.

Também foi realizada uma avaliação por **fatias de dados**, onde o desempenho do modelo foi medido separadamente para diferentes valores de variáveis categóricas (por exemplo, diferentes tipos de trabalho, níveis de educação, sexo, etc.).

---

## Metrics

As seguintes métricas foram utilizadas para avaliar o modelo:

- **Precision (Precisão)**: mede quantas previsões positivas do modelo estavam corretas.
- **Recall**: mede quantos dos casos positivos reais foram corretamente identificados pelo modelo.
- **F1-score (F-beta com beta=1)**: média harmônica entre precisão e recall.

Desempenho geral do modelo no conjunto de teste (exemplo):

- **Precision:** 0.73
- **Recall:** 0.63
- **F1-score:** 0.68

Além disso, as métricas foram calculadas para diferentes grupos da população (fatias dos dados), e os resultados foram armazenados no arquivo `slice_output.txt`. Foi observado que o desempenho varia entre grupos, o que pode indicar possíveis desigualdades no comportamento do modelo.

---

## Ethical Considerations

Este modelo utiliza dados sensíveis como sexo, raça e país de origem. Isso significa que ele pode aprender padrões que refletem desigualdades presentes na sociedade.

Existe o risco de:

- O modelo ter desempenho diferente entre grupos
- Reforçar vieses existentes nos dados
- Produzir previsões menos precisas para certos grupos

Por isso, ele **não deve ser usado em aplicações reais** que impactem diretamente pessoas.

---

## Caveats and Recommendations

- O modelo foi treinado com um conjunto de dados específico e pode não generalizar bem para outras populações.
- As previsões dependem fortemente da qualidade e representatividade dos dados.
- O modelo é relativamente simples e não foi profundamente ajustado.
- Recomenda-se sempre avaliar o desempenho por grupos (análise de fatias) antes de qualquer uso.
- Para aplicações reais, seriam necessários testes adicionais, análise de viés e validação ética.
