import pandas as pd
import re
from nltk.corpus import stopwords

def get_df(query: str):
    try:
        df = pd.read_csv(f'./relevants/{query}_relevant.csv')
    except:
        df = pd.DataFrame()

    if 'Unnamed: 0' not in df.columns:
        df['Unnamed: 0'] = []
    if 'ID' not in df.columns:
        df['ID'] = []

    return df

def query_processing (query):
    query=re.sub('\W', ' ', query)
    query = query.strip().lower()
    query = " ".join([word for word in query.split() if word not in stopwords.words('english')])
    return query