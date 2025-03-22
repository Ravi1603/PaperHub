from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.hooks.postgres_hook import PostgresHook
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import time

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fetch_data_from_arxiv',
    default_args=default_args,
    description='A DAG to fetch data from arxiv and store in PostgreSQL',
    schedule_interval=timedelta(days=1),
)

def fetch_arxiv_data(search_query, max_results=2000, results_per_iteration=100):
    base_url = 'http://export.arxiv.org/api/query?'
    all_papers = []
    
    for start in range(0, max_results, results_per_iteration):
        print(f'Fetching results {start} to {start + results_per_iteration} for query "{search_query}"')
        query = f'search_query=all:{search_query}&start={start}&max_results={results_per_iteration}'
        response = requests.get(base_url + query)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = {
                    'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
                    'authors': [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
                    'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text,
                    'category': entry.find('{http://arxiv.org/schemas/atom}primary_category').attrib['term'],
                    'id': entry.find('{http://www.w3.org/2005/Atom}id').text
                }
                all_papers.append(paper)
            time.sleep(3)  # Be nice to the arXiv API and wait between requests
        else:
            print(f'Error: {response.status_code}')
            break
    return all_papers

def transform_data(papers):
    df_arxiv = pd.json_normalize(papers)

    # Select relevant columns for arXiv
    df_arxiv = df_arxiv[['id', 'title', 'authors', 'abstract', 'category']]
    df_arxiv['dataset'] = 'arxiv'

    # Add random citations for arXiv data
    if 'citations' not in df_arxiv.columns:
        df_arxiv['citations'] = np.random.randint(1, 6, size=len(df_arxiv))
    
    return df_arxiv

def store_papers_in_db(df_papers):
    pg_hook = PostgresHook(postgres_conn_id='papers_connection')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO arxiv_papers_table (id, title, authors, abstract, category, dataset, citations)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING;
    """
    
    for _, row in df_papers.iterrows():
        cursor.execute(insert_query, (row['id'], row['title'], row['authors'], row['abstract'], row['category'], row['dataset'], row['citations']))
    
    conn.commit()
    cursor.close()
    conn.close()

def fetch_transform_store():
    search_queries = [
        'machine learning',
        'big data',
        'cloud computing',
        'deep learning',
        'adversarial attacks',
        'image classification',
        'image segmentation',
        'federated learning',
        'computer vision',
        'AI'
    ]
    
    all_papers = []
    for query in search_queries:
        papers = fetch_arxiv_data(query)
        all_papers.extend(papers)
    
    df_papers = transform_data(all_papers)
    store_papers_in_db(df_papers)

create_table_task = SQLExecuteQueryOperator(
    task_id='create_arxiv_papers_table',
    conn_id='papers_connection',
    sql="""
    CREATE TABLE IF NOT EXISTS arxiv_papers_table (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        authors TEXT[] NOT NULL,
        abstract TEXT NOT NULL,
        category TEXT NOT NULL,
        dataset TEXT NOT NULL,
        citations INTEGER NOT NULL
    );
    """,
    dag=dag,
)

fetch_transform_store_task = PythonOperator(
    task_id='fetch_transform_store',
    python_callable=fetch_transform_store,
    dag=dag,
)

create_table_task >> fetch_transform_store_task