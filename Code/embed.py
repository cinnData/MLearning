# Resources #
import pandas as pd, numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Importing the data #
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'covid_abstracts.csv')
df.head()
pd.concat([df['title'].str.len().describe(), df['abstract'].str.len().describe()], axis=1)

# Encoding the titles #
model = SentenceTransformer('all-distilroberta-v1')
titles = df['title'].to_list()
title_embed = model.encode(titles, output_value='sentence_embedding', convert_to_numpy=True)
df['title_embed'] = title_embed.tolist()

# Encoding the queries #
queries = ['persistent COVID']
query_embed = model.encode(queries, output_value='sentence_embedding', convert_to_numpy=True).tolist()

# Semantic search #
sim = df['title_embed'].apply(lambda x: cosine(x, query_embed[0]))
sim.name = 'similarity'
title_search_output = pd.concat([df['title'], sim], axis=1).sort_values(by='similarity').head(5)

# Encoding the abstracts #
abstracts = df['abstract'].to_list()
abstract_embed = model.encode(abstracts, output_value='sentence_embedding', convert_to_numpy=True)
df['abstract_embed'] = abstract_embed.tolist()

# Semantic search #
sim = df['abstract_embed'].apply(lambda x: cosine(x, query_embed[0]))
sim.name = 'similarity'
abstract_search_output = pd.concat([df['title'], sim], axis=1).sort_values(by='similarity').head(5)
