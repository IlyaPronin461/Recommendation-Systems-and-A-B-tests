# uvicorn app:app --reload --port 8899 для загрузки на сервер

import logging
from fastapi import FastAPI, HTTPException
from datetime import datetime
from typing import List
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine
import os
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostGet(BaseModel):
    id: int
    text: str
    topic: str


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
    logger.info(f"Загружены данные по запросу: {query}")
    return pd.concat(chunks, ignore_index=True)


def load_features() -> pd.DataFrame:
    users_features = batch_load_sql('select * from public.ilja_pronin_gfr6897_lesson22_step_8_user_data_1')
    posts_features = batch_load_sql('select * from public.ilja_pronin_gfr6897_lesson22_step_7_new_post_text_df_1')
    liked_posts = batch_load_sql("SELECT distinct post_id, user_id FROM public.feed_data where action='like'")
    logger.info("Фичи загружены успешно")
    return users_features, posts_features, liked_posts


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("catboost_model_step_8")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    logger.info("Модель загружена успешно")
    return loaded_model


app = FastAPI()

model = load_models()
user_data, post_data, liked_posts = load_features()


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    try:
        logger.info(
            f"Получен запрос на рекомендации для пользователя {id} с параметром времени {time} и лимитом {limit}")

        user_by_id = user_data[user_data['user_id'] == id].copy()
        user_by_id = user_by_id.drop('user_id', axis=1)

        add_user_data = dict(zip(user_by_id.columns, user_by_id.values[0]))
        posts_and_users = post_data.assign(**add_user_data)

        posts_and_users["day"] = time.day
        posts_and_users["month"] = time.month

        predicts = model.predict_proba(posts_and_users[['gender', 'age', 'country', 'city', 'exp_group', 'os', 'source',
       'is_teenagers', 'is_youth', 'is_adults', 'have_million_people',
       'is_capital', 'city_mean_age', 'city_median_age', 'ratio_0', 'ratio_1',
       'is_russia', 'city_mean_exp_group', 'age_city_mean_age',
       'age_city_median_age', 'city_mean_age_city_median_age', 'text', 'topic',
       'cluster', 'pca_1', 'pca_2', 'text_length', 'word_count',
       'day', 'month']])[:, 1]

        posts_and_users["predicts"] = predicts

        top_recommended_posts = posts_and_users[~posts_and_users.index.isin(liked_posts)]
        top_recommended_posts = top_recommended_posts.sort_values('predicts', ascending=False).head(limit)

        response = [
            PostGet(
                id=row['post_id'],
                text=row['text'],
                topic=row['topic']
            )
            for _, row in top_recommended_posts.iterrows()
        ]

        logger.info(f"Рекомендовано {len(response)} постов для пользователя {id}")
        return response

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
