import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import unidecode
import freeman as fm
import io
import networkx as nx
from itertools import combinations
from scipy.stats import ttest_ind



######### loading the dataset #########
netflix_overall=pd.read_csv("netflix_titles.csv")

netflix_movies=netflix_overall[netflix_overall['type']=='Movie']

netflix_movies = netflix_movies.dropna(axis=0)
cast = netflix_movies['cast']


actors = []
temp = []


for i in cast:
    temp = i.split(",")
    for name in temp:
        actors.append(name)
        
actors_unique = Counter(actors).keys()  
print("atores sem repetição:",len(actors_unique))
print("atores com repetição:", len(actors))
#Counter(actors).values()

netflix_movies = netflix_movies.reset_index(drop=True)

######### Adicionando colunas indicando o filme adicionado no ano ##########

netflix_movies['added_2017'] = False
netflix_movies['added_2018'] = False
netflix_movies['added_2019'] = False
for i in range(0, len(netflix_movies)):
    if (netflix_movies['date_added'][i][-4:] == '2017'):
        netflix_movies['added_2017'][i] = True
    else:
        netflix_movies['added_2017'][i] = False

for i in range(0, len(netflix_movies)):
    if (netflix_movies['date_added'][i][-4:] == '2018'):
        netflix_movies['added_2018'][i] = True
    else:
        netflix_movies['added_2018'][i] = False

for i in range(0, len(netflix_movies)):
    if (netflix_movies['date_added'][i][-4:] == '2019'):
        netflix_movies['added_2019'][i] = True
    else:
        netflix_movies['added_2019'][i] = False

######### Separando os gêneros por ano => variável GMP #########

netflix_movies_2017 = netflix_movies[netflix_movies.added_2017 == True]
netflix_movies_2018 = netflix_movies[netflix_movies.added_2018 == True]
netflix_movies_2019 = netflix_movies[netflix_movies.added_2019 == True]

genre_2017 = netflix_movies_2017['listed_in']
genre_count_2017 = pd.Series(dict(Counter(','.join(genre_2017).replace(' ,',',').replace(', ',',')
                                    .split(',')))).sort_values(ascending=False)

genre_2018 = netflix_movies_2018['listed_in']
genre_count_2018 = pd.Series(dict(Counter(','.join(genre_2018).replace(' ,',',').replace(', ',',')
                                    .split(',')))).sort_values(ascending=False)

genre_2019 = netflix_movies_2019['listed_in']
genre_count_2019 = pd.Series(dict(Counter(','.join(genre_2019).replace(' ,',',').replace(', ',',')
                                    .split(',')))).sort_values(ascending=False)

actors_and_genres2017 = netflix_movies_2017.iloc[:, [4, 10]]
actors_and_genres2018 = netflix_movies_2018.iloc[:, [4, 10]]
actors_and_genres2019 = netflix_movies_2019.iloc[:, [4, 10]]


######### Fazendo um dataframe com atores e suas respectivas categorias #########

network_2017_temp_actor = pd.concat([pd.Series(row['cast'], row['listed_in'].split(','))              
                    for _, row in actors_and_genres2017.iterrows()]).reset_index()

network_2017_temp_actor = network_2017_temp_actor.rename(columns={'index': 'categoria', 0: 'ator'})

network_2017_actor = pd.concat([pd.Series(row["categoria"], row['ator'].split(','))              
                    for _, row in network_2017_temp_actor.iterrows()]).reset_index()

network_2017_actor = network_2017_actor.rename(columns={'index': 'ator', 0: 'categoria'})

# agrupando categorias por ator
actor_genre_2017 = network_2017_actor.groupby('ator')['categoria'].apply(list).reset_index(name='genero')

######### GA, EA e generos por ator #########

actor_2017 = netflix_movies_2017['cast']
actor_count_2017 = pd.Series(dict(Counter(','.join(actor_2017).replace(' ,',',').replace(', ',',')
                                    .split(',')))).sort_values(ascending=False)

actor_count_2017 = actor_count_2017.to_frame()

actor_count_2017.reset_index(level = 0, inplace = True)
actor_count_2017 = actor_count_2017.rename(columns={'index': 'ator', 0: 'GA'})

actor_count_2017['EA'] = actor_count_2017['GA'].apply(lambda x: 1/x)


variables = actor_count_2017.set_index('ator').join(actor_genre_2017.set_index('ator'))

variables.dropna()

# separando ator por categoria
network_2017_temp = pd.concat([pd.Series(row['listed_in'], row['cast'].split(','))              
                    for _, row in actors_and_genres2017.iterrows()]).reset_index()

network_2017_temp = network_2017_temp.rename(columns={'index': 'ator', 0: 'categoria'})

network_2017 = pd.concat([pd.Series(row["ator"], row['categoria'].split(','))              
                    for _, row in network_2017_temp.iterrows()]).reset_index()

network_2017 = network_2017.rename(columns={'index': 'categoria', 0: 'ator'})

######### limpando os nomes dos atores #########
for i in range(len(network_2017['ator'])):
    if "'" in network_2017['ator'][i]:
        network_2017['ator'][i] = network_2017['ator'][i].replace("'", "")
    if '"' in network_2017['ator'][i]:
        network_2017['ator'][i] = network_2017['ator'][i].replace('"', "")
        
    network_2017['ator'][i] = unidecode.unidecode(network_2017['ator'][i])


######### fazendo o arquivo .gml para fazer a rede #########

nodes1 = list(set(network_2017['ator']))
nodes2 = list(set(network_2017['categoria']))

temp_cat = []
temp_ator = []
edgeCheck = {}
listweights = []

with io.open("./network2.gml", "w") as f:
    f.write('graph [\n')
    f.write('  directed 0\n')

    for i in range(len(network_2017['ator'])):
        if not network_2017['ator'][i] in temp_ator:
            f.write('  node [\n')
            f.write('    id "{}"\n'.format(network_2017['ator'][i]))
            f.write('  ]\n')
            temp_ator.append(network_2017['ator'][i])
        
    for i in range(len(network_2017['categoria'])):
        if not network_2017['categoria'][i] in temp_cat:
            f.write('  node [\n')
            f.write('    id "{}"\n'.format(network_2017['categoria'][i]))
            f.write('  ]\n')
            temp_cat.append(network_2017['categoria'][i])
            
    for index, row in network_2017.iterrows():
        keytemp = row['ator'] + ',' + row['categoria']
        if keytemp in edgeCheck:
            edgeCheck[keytemp] += 1
        else:
            edgeCheck[keytemp] = 1
            
    for key in edgeCheck:
        ator, categoria = key.split(',')
        weight = edgeCheck[key]
        listweights.append(weight)
        
        f.write('  edge [\n')
        f.write('    source "{}"\n'.format(ator))
        f.write('    target "{}"\n'.format(categoria))
        f.write('    weight {}\n'.format(weight))
        f.write('  ]\n')  

    f.write(']\n')

g = fm.load('network2.gml')
g.label_nodes()
g.set_all_nodes(size=5, labpos='hover')
g.set_all_edges(color=(0, 0, 0, 0.5))
#g.draw()

#centralidade degree dos nós
nx.degree_centrality(g)
nx.density(g)
        
def load():
    g = fm.load('network2.gml')

    # Remover todas as arestas com peso menor ou igual a 0.5.
    # Precisamos de dois loops, pois não é uma boa ideia
    # tirar algo de um conjunto enquanto iteramos nele.
    removed = []
    for n, m in g.edges:
        if g.edges[n, m]['weight'] <= 2:
            removed.append((n, m))
    for n, m in removed:
        g.remove_edge(n, m)

    # Remover todos os nós que ficaram isolados depois da
    # remoção das arestas, para melhorar a visualização.
    removed = []
    for n in g.nodes:
        if not g.degree(n):
            removed.append(n)
    for n in removed:
        g.remove_node(n)

    return g

######### transformando em one-mode #########

#Construindo a rede one-mode do zero:
g1 = fm.Graph(nx.Graph())   

length = len(temp_ator)
middle_index = length//10

first_half = temp_ator[:middle_index]

#Adicionando todos os nós de usuário à rede one-mode:
for n in first_half:
    g1.add_node(n)

for n, m in combinations(g1.nodes, 2):

    # Muito cuidado para não usar g1 aqui!
    # Estamos analisando os vizinhos em g2.
    repos_n = set(g.neighbors(n))
    repos_m = set(g.neighbors(m))

    # Em sets é fácil calcular intersecção.
    weight = len(repos_n & repos_m)

    # Adicionamos só se weight for positivo.
    if weight > 0:
        g1.add_edge(n, m)
        g1.edges[n, m]['weight'] = weight

isolated = [n for n in g1.nodes if g1.degree(n) == 0]

for n in isolated:
    g1.remove_node(n)

weights = [g1.edges[n, m]['weight'] for n, m in g1.edges]

nx.write_gml(g1,'one_mode.gml')

######### Métricas #########

# CA (r) 

cluster = nx.clustering(g1)

for n in g1.nodes:
    for m in g1.neighbors(n):
        if cluster[n] == 1:
            print(n, m)

######### Teste de hipótese (teste-t) ########

categories = network_2017['categoria'].unique()
list_cat = []

print(network_2017['categoria'].unique())

cat1 = network_2017['ator']
cat2 = network_2017['categoria']

print(cat2)

ttest_ind(100, cat2)