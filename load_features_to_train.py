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
    SELECT u.*, p.*, f.timestamp, f.action, f.target
    FROM public.feed_data f
    INNER JOIN public.ilja_pronin_gfr6897_dl_lesson10_step_5_user_data u ON f.user_id = u.user_id
    INNER JOIN public.ilja_pronin_gfr6897_dl_lesson10_step_5_new_post_text_df p ON f.post_id = p.post_id
    LIMIT {limit} OFFSET {offset};
    """

    data = batch_load_sql(query)

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month

    # дополняем
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)  # 5 - суббота, 6 - воскресенье

    # Категоризация времени суток
    def get_time_of_day(hour):
        if 12 <= hour < 18:
            return 1
        else:
            return 0

    data['is_afternoon'] = data['hour'].apply(get_time_of_day)

    # удаляем сильно коррелирующие и ненужные столбцы
    data = data.drop(
        ['action', 'user_id', 'post_id', 'timestamp', 'text'], axis=1)

    return data