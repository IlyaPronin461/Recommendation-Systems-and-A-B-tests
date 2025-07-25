# в этом файле я качаю таблицу post_text_df, генерю там новые фитчи и загружаю в новую БД
# public.ilja_pronin_gfr6897_lesson22_step_7_new_post_text_df_1

import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_post_text_df() -> pd.DataFrame:
    query = """
             SELECT * 
             FROM public.post_text_df;
             """

    return batch_load_sql(query)


new_post_text_df = load_post_text_df()

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(new_post_text_df['text'])

kmeans = KMeans(n_clusters=5, random_state=42)
new_post_text_df['cluster'] = kmeans.fit_predict(X_tfidf)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_tfidf.toarray())

new_post_text_df['pca_1'] = pca_components[:, 0]
new_post_text_df['pca_2'] = pca_components[:, 1]

new_post_text_df['text_length'] = new_post_text_df['text'].apply(len)
new_post_text_df['word_count'] = new_post_text_df['text'].apply(lambda x: len(x.split()))

new_post_text_df.to_sql('ilja_pronin_gfr6897_lesson22_step_7_new_post_text_df_1', con=engine, if_exists='replace', index=False)

