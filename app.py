import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

anime_dataset = pd.read_csv("./data/anime-dataset-2023.csv")
anime_2020_dataset = pd.read_csv("./data/anime-filtered.csv")
ax = plt.axes()
sns.heatmap(anime_dataset.isna().transpose(), cbar=False, ax=ax)
plt.xlabel("Columns")
plt.ylabel("Missing Values")
column_of_interest = ["Name"]
anime_dataset["English name"] = (
    np.where(anime_dataset["English name"] == "UNKNOWN", anime_dataset["Name"], anime_dataset["English name"])
)
anime_dataset["Scored By"] = (
    np.where(anime_dataset["Scored By"] == "UNKNOWN", 0, anime_dataset["Scored By"]))
print(anime_dataset["English name"])