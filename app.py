# public.ilja_pronin_gfr6897_lesson22_step_8_user_data_1 - user features in ML block; model_control
# public.ilja_pronin_gfr6897_lesson22_step_7_new_post_text_df_1 - post features in ML block; model_control

# public.ilja_pronin_gfr6897_dl_lesson10_step_5_user_data - user features in DL block; test
# public.ilja_pronin_gfr6897_dl_lesson10_step_5_new_post_text_df - post features in DL block; test

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
import hashlib


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


salt = 'my_salt'
threshold = 50

#Функция для определения групп для A/B тестирования
def get_exp_group(user_id: int) -> str:
    hash_value = int(hashlib.md5((str(user_id) + salt).encode()).hexdigest(), 16)
    return 'control' if (hash_value % 100) < threshold else 'test'


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


def load_features_control() -> pd.DataFrame:
    users_features = batch_load_sql('select * from public.ilja_pronin_gfr6897_lesson22_step_8_user_data_1')
    posts_features = batch_load_sql('select * from public.ilja_pronin_gfr6897_lesson22_step_7_new_post_text_df_1')
    logger.info("Фичи control загружены успешно")
    return users_features, posts_features


def load_features_test() -> pd.DataFrame:
    users_features = batch_load_sql('select * from public.ilja_pronin_gfr6897_dl_lesson10_step_5_user_data')
    posts_features = batch_load_sql('select * from public.ilja_pronin_gfr6897_dl_lesson10_step_5_new_post_text_df')
    logger.info("Фичи test загружены успешно")
    return users_features, posts_features


def get_model_path(model_version: str) -> str:
    if (
        os.environ.get("IS_LMS") == "1"
    ):
        model_path = f"/workdir/user_input/model_{model_version}"
    else:
        model_path = f"model_{model_version}"
    return model_path


def load_models(model_version: str):
    model_path = get_model_path(model_version)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    logger.info("Модель загружена успешно")
    return loaded_model



app = FastAPI()


model_test = load_models('test')
model_control = load_models('control')

user_data_control, post_data_control = load_features_control()
user_data_test, post_data_test = load_features_test()
liked_posts = batch_load_sql("SELECT distinct post_id, user_id FROM public.feed_data where action='like'")
logger.info("Фичи liked_posts загружены успешно, можно делать запрос")

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    try:
        logger.info(
            f"Получен запрос на рекомендации для пользователя {id} с параметром времени {time} и лимитом {limit}")

        exp_group = get_exp_group(id)

        if exp_group == 'control':
            user_by_id = user_data_control[user_data_control['user_id'] == id].copy()
            user_by_id = user_by_id.drop('user_id', axis=1)

            add_user_data = dict(zip(user_by_id.columns, user_by_id.values[0]))
            posts_and_users = post_data_control.assign(**add_user_data)

            posts_and_users["day"] = time.day
            posts_and_users["month"] = time.month

            predicts = model_control.predict_proba(
                posts_and_users[['gender', 'age', 'country', 'city', 'exp_group', 'os', 'source',
                                 'is_teenagers', 'is_youth', 'is_adults', 'have_million_people',
                                 'is_capital', 'city_mean_age', 'city_median_age', 'ratio_0', 'ratio_1',
                                 'is_russia', 'city_mean_exp_group', 'age_city_mean_age',
                                 'age_city_median_age', 'city_mean_age_city_median_age', 'text', 'topic',
                                 'cluster', 'pca_1', 'pca_2', 'text_length', 'word_count',
                                 'day', 'month']])[:, 1]

            posts_and_users["predicts"] = predicts

            top_recommended_posts = posts_and_users[~posts_and_users.index.isin(liked_posts)]
            top_recommended_posts = top_recommended_posts.sort_values('predicts', ascending=False).head(limit)

            recommendation = [
                PostGet(
                    id=row['post_id'],
                    text=row['text'],
                    topic=row['topic']
                )
                for _, row in top_recommended_posts.iterrows()
            ]

            logger.info(f"Рекомендовано {len(recommendation)} постов для пользователя {id}")

        elif exp_group == 'test':

            user_by_id = user_data_test[user_data_test['user_id'] == id].copy()
            user_by_id = user_by_id.drop('user_id', axis=1)

            add_user_data = dict(zip(user_by_id.columns, user_by_id.values[0]))
            posts_and_users = post_data_test.assign(**add_user_data)

            posts_and_users["day"] = time.day
            posts_and_users["month"] = time.month

            posts_and_users["day_of_week"] = time.weekday()

            posts_and_users['hour'] = time.hour
            posts_and_users['is_weekend'] = posts_and_users['day_of_week'].isin([5, 6]).astype(
                int)  # 5 - суббота, 6 - воскресенье

            def get_time_of_day(hour):
                if 12 <= hour < 18:
                    return 1
                else:
                    return 0

            posts_and_users['is_afternoon'] = posts_and_users['hour'].apply(get_time_of_day)

            predicts = model_test.predict_proba(
                posts_and_users[['gender', 'age', 'country', 'city', 'exp_group', 'os', 'source',
                                 'is_teenagers', 'is_youth', 'is_adults', 'have_million_people',
                                 'is_capital', 'city_mean_age', 'city_median_age', 'ratio_1',
                                 'is_russia', 'city_mean_exp_group', 'city_popularity', 'topic',
                                 'avg_word_length', 'unique_word_count', 'pca_roberta_0',
                                 'pca_roberta_1', 'pca_roberta_2', 'pca_roberta_3', 'pca_roberta_4',
                                 'pca_roberta_5', 'pca_roberta_6', 'pca_roberta_7', 'pca_roberta_8',
                                 'pca_roberta_9', 'pca_roberta_10', 'pca_roberta_11', 'pca_roberta_12',
                                 'pca_roberta_13', 'pca_roberta_14', 'pca_roberta_15', 'pca_roberta_16',
                                 'pca_roberta_17', 'pca_roberta_18', 'pca_roberta_19', 'pca_roberta_20',
                                 'pca_roberta_21', 'pca_roberta_22', 'pca_roberta_23', 'pca_roberta_24',
                                 'pca_roberta_25', 'pca_roberta_26', 'pca_roberta_27', 'pca_roberta_28',
                                 'pca_roberta_29', 'pca_roberta_30', 'pca_roberta_31', 'day',
                                 'month', 'hour', 'day_of_week', 'is_weekend', 'is_afternoon']])[:, 1]

            posts_and_users["predicts"] = predicts

            top_recommended_posts = posts_and_users[~posts_and_users.index.isin(liked_posts)]
            top_recommended_posts = top_recommended_posts.sort_values('predicts', ascending=False).head(limit)

            recommendation = [
                PostGet(
                    id=row['post_id'],
                    text=row['text'],
                    topic=row['topic']
                )
                for _, row in top_recommended_posts.iterrows()
            ]

            logger.info(f"Рекомендовано {len(recommendation)} постов для пользователя {id}")

        else:
            raise ValueError('Неизвестная группа')

        return Response(exp_group=exp_group, recommendations=recommendation)


    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


