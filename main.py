import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np

uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'

dados = pd.read_csv(uri)
print(dados.head(3))

novo_nome_colunas = {
    'unfinished': 'nao_finalizado',
    'expected_hours': 'horas_esperadas',
    'price': 'preco'
}
dados = dados.rename(columns=novo_nome_colunas)
print(dados.head(3))

troca = {
    0:1,
    1:0
}
dados['finalizados'] = dados.nao_finalizado.map(troca)
dados.head()
#dados.tail()

sns.scatterplot(x="horas_esperadas", y="preco", data=dados)
print('Grafico geral Horas x Preço, plotando os finalizados')

sns.scatterplot(x="horas_esperadas", y="preco", data=dados, hue='finalizados')
print('Grafico geral Horas x Preço, plotando os finalizados COLORIDOS')

sns.relplot(x="horas_esperadas", y="preco", data=dados, col='finalizados')
print('Grafico geral Horas x Preço, plotando os finalizados.','\n',
'Formato RELATIVO'
)

sns.relplot(
    x="horas_esperadas", y="preco", data=dados, col='finalizados', hue='finalizados'
)
print('Grafico geral Horas x Preço, plotando os finalizados.','\n',
'Formato RELATIVO e COLORIDO'
)


x = dados[['horas_esperadas','preco']]
y = dados['finalizados']

SEED = 20
np.random.seed(SEED) ### Se a Lib de dados usar o np como base. Nesse caso a sklearn usa.

treino_x, teste_x, treino_y, teste_y = train_test_split(
    x, y, test_size = 0.25, stratify=y
)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

base_line = np.ones(540)
acuracia_base_line = accuracy_score(teste_y, base_line) * 100
print("A acurácia da Baseline foi %.2f%%" % acuracia_base_line)

qtd_dados_treino = treino_y.value_counts()
proporcao_treino_negativos = qtd_dados_treino[0]
proporcao_treino_positivos = qtd_dados_treino[1]
proporcao_treino = proporcao_treino_negativos/proporcao_treino_positivos

qtd_dados_teste = teste_y.value_counts()
proporcao_teste_negativos = qtd_dados_teste[0]
proporcao_teste_positivos = qtd_dados_teste[1]
proporcao_teste = proporcao_teste_negativos/proporcao_teste_positivos

print(f'Proporção dos testes: {proporcao_teste}',
      f'Proporção dos treinos: {proporcao_treino}', sep='\n')