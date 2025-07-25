import pandas as pd
from sqlalchemy import create_engine


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features(limit: int, offset: int = 0) -> pd.DataFrame:
    query = f"""
    SELECT f.timestamp, u.gender, u.age, u.country, u.city, u.exp_group, u.os, u.source, u.is_teenagers, u.is_youth, u.is_adults, u.have_million_people, u.is_capital, u.city_mean_age, u.city_median_age, u.ratio_0, u.ratio_1, u.is_russia, u.city_mean_exp_group, u.age_city_mean_age, u.age_city_median_age, u.city_mean_age_city_median_age, p.text, p.topic, p.cluster, p.pca_1, p.pca_2, p.text_length, p.word_count, f.target
    FROM public.feed_data f
    INNER JOIN public.ilja_pronin_gfr6897_lesson22_step_8_user_data_1 u ON f.user_id = u.user_id
    INNER JOIN public.ilja_pronin_gfr6897_lesson22_step_7_new_post_text_df_1 p ON f.post_id = p.post_id
    LIMIT {limit} OFFSET {offset};
    """

    data = batch_load_sql(query)

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month
    data = data.drop('timestamp', axis=1)

    return data