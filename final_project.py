import os
import streamlit as st
import pandas as pd
from utils import get_df
from information_retrieval_system_class import InformationRetrievalSystem

if not os.path.exists('./relevants'):
    os.makedirs('./relevants')

df = pd.read_csv('./classes.csv')

major_description_model = InformationRetrievalSystem(df, 'Major_Course_Description')
title_description_model = InformationRetrievalSystem(df, 'Course Title_Course_Description')
major_title_model = InformationRetrievalSystem(df, 'Major_Course_Title')
major_title_description_model = InformationRetrievalSystem(df, 'Major_Course_Title_Course_Description')

st.title('Information Retrieval System')
query = st.text_input('insert your query:', placeholder='example: computer science')

quantity = st.slider('number of results:', 1, 100, 10)

major_description_model_search = major_description_model.search(query, topn=100)

if major_description_model_search.shape[0] > 0:
    df = get_df(query)
    c = st.container()
    c.write('results:')
    for i in range(major_description_model_search.shape[0]):
        expander = st.expander(f'{major_description_model_search.iloc[i]["Course Title"]}')
        expander.write('major: {}'.format(major_description_model_search.iloc[i]['Major']))
        expander.write('description: {}'.format(major_description_model_search.iloc[i]['Course Description_Clean']))
        value = major_description_model_search.iloc[i]['ID'] in df['ID'].values
        agree = expander.checkbox('is it relevant?', key=i, value=value)
        if agree:
            if major_description_model_search.iloc[i]['ID'] in df['ID'].values:
                expander.write('already marked as relevant')
            else:
                expander.write('thanks!')
                new_row = {
                    'Unnamed: 0': len(df),
                    'ID': major_description_model_search.iloc[i]['ID']
                }
                df = df._append(new_row, ignore_index=True)
                df.to_csv(f'./relevants/{query}_relevants.csv', index=False)