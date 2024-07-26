import pandas as pd
import numpy as np
from utils import query_processing
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class InformationRetrievalSystem:
    def __init__(self, df: pd.DataFrame, col: str) -> None:
        self.df = df
        self.col = col
        self._get_tfidf_vectors()

    def _get_tfidf_vectors(self) -> None:
        self.vectorizer = TfidfVectorizer()
        self.tfidf_vectors = self.vectorizer.fit_transform(self.df[self.col])

    def _get_topn_results(self, scores: np.ndarray[Any, Any], topn: int) -> pd.DataFrame:
        df2 = self.df.copy()
        df2['score'] = scores

        sorted_indices = scores.argsort()[::-1]
        return df2.iloc[sorted_indices[:topn]][['ID', 'Major', 'Course Title', 'Course Description_Clean', 'score']]

    def search(self, query: str, topn:int=10) -> pd.DataFrame:
        query = query_processing(query)
        if query == '':
            return pd.DataFrame()

        query_vectorized = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vectorized, self.tfidf_vectors).flatten()
        return self._get_topn_results(scores, topn)