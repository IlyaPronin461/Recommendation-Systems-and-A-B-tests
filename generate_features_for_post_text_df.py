# в этом файле я качаю таблицу post_text_df, генерю там новые фитчи и загружаю в новую БД
# public.ilja_pronin_gfr6897_dl_lesson10_step_5_new_post_text_df

from warnings import filterwarnings

filterwarnings('ignore')

import pandas as pd
from sqlalchemy import create_engine
from sklearn.decomposition import PCA

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


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

new_post_text_df['text_length'] = new_post_text_df['text'].apply(len)
new_post_text_df['word_count'] = new_post_text_df['text'].apply(lambda x: len(x.split()))
new_post_text_df['avg_word_length'] = new_post_text_df['text'].apply(
    lambda x: np.mean([len(word) for word in x.split()]))
new_post_text_df['unique_word_count'] = new_post_text_df['text'].apply(lambda x: len(set(x.split())))

# начинаем использовать deep learning для генерирования новых признаков
# Загрузка предобученной модели RoBERTa
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def get_roberta_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


roberta_embeddings = get_roberta_embeddings(new_post_text_df['text'].tolist())

pca = PCA(n_components=32)
pca_embeddings = pca.fit_transform(roberta_embeddings)

pca_columns = [f'pca_roberta_{i}' for i in range(pca_embeddings.shape[1])]
pca_df = pd.DataFrame(pca_embeddings, columns=pca_columns)

new_post_text_df = pd.concat([new_post_text_df, pca_df], axis=1)

new_post_text_df = new_post_text_df.drop(['word_count', 'text_length'], axis=1)

new_post_text_df.to_sql('ilja_pronin_gfr6897_dl_lesson10_step_5_new_post_text_df', con=engine, if_exists='replace', index=False)