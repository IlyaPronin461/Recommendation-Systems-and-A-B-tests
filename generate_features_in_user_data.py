# в этом файле я качаю таблицу user_data, генерю там новые фитчи и загружаю в новую БД
# public.ilja_pronin_gfr6897_lesson22_step_8_user_data_1


import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import PolynomialFeatures


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


def load_user_data() -> pd.DataFrame:
    query = """
             SELECT * 
             FROM public.user_data;
             """

    return batch_load_sql(query)


new_user_data = load_user_data()

# age cats
new_user_data['is_teenagers'] = ((new_user_data['age'] >= 13) & (new_user_data['age'] <= 17)).astype(int)
new_user_data['is_youth'] = ((new_user_data['age'] >= 18) & (new_user_data['age'] <= 35)).astype(int)
new_user_data['is_adults'] = ((new_user_data['age'] >= 36) & (new_user_data['age'] <= 64)).astype(int)

# city cats
city_list = [
  "Moscow",
  "Saint Petersburg",
  "Novosibirsk",
  "Kazan",
  "Yekaterinburg",
  "Chelyabinsk",
  "Samara",
  "Omsk",
  "Rostov",
  "Ufa",
  "Volgograd",
  "Perm",
  "Krasnoyarsk",
  "Voronezh",
  "Saratov"
]
new_user_data['have_million_people'] = new_user_data['city'].apply(lambda x: 1 if x in city_list else 0)
capital_list = [
    "Moscow",
    "Minsk",
    "Kiev",
    "Saint Petersburg",
    "Baku",
    "Tallinn",
    "Riga",
    "Vilnius",
    "Helsinki",
    "Warsaw",
    "Berlin",
    "Prague",
    "Vienna",
    "Budapest",
    "Rome",
    "Paris",
    "Madrid",
    "London",
    "Ankara",
    "Istanbul",
    "Kyiv",
    "Nicosia",
    "Athens",
    "Belgrade",
    "Sofia",
    "Bern",
    "Zurich",
    "Lisbon",
    "Oslo",
    "Copenhagen",
    "Stockholm",
    "Amsterdam",
    "Brussels",
    "Luxembourg",
    "Reykjavik",
    "Cardiff",
    "Edinburgh",
    "Dublin",
    "Lisburn",
    "Gibraltar",
    "Malta",
    "Valetta",
    "Larnaca",
    "Limassol",
    "Paphos",
    "Chisinau",
    "Tbilisi",
    "Yerevan",
    "Astana",
    "Almaty",
    "Tashkent",
    "Bishkek",
    "Dushanbe",
    "Ashgabat",
    "Minsk",
    "Tirana",
    "Andorra la Vella",
    "Monaco",
    "San Marino",
    "Vatican City"
]
new_user_data['is_capital'] = new_user_data['city'].apply(lambda x: 1 if x in capital_list else 0)
# Расчет среднего возраста для каждого города
city_avg_age = new_user_data.groupby('city')['age'].mean().to_dict()
new_user_data['city_mean_age'] = new_user_data['city'].map(city_avg_age)
# Расчет медианного возраста для каждого города
country_median_age = new_user_data.groupby('city')['age'].median().to_dict()
new_user_data['city_median_age'] = new_user_data['city'].map(country_median_age)
# Распределение пользователей по полу в городе
city_gender_ratio = new_user_data.groupby('city')['gender'].value_counts(normalize=True).unstack().fillna(0)
new_user_data = new_user_data.merge(city_gender_ratio, left_on='city', right_index=True, how='left')
new_user_data = new_user_data.rename(columns={
    0: 'ratio_0',
    1: 'ratio_1'
})


# os cats
new_user_data['os'] = new_user_data['os'].apply(lambda x: 1 if x == 'Android' else 0)

# source cats
new_user_data['source'] = new_user_data['source'].apply(lambda x: 1 if x == 'ads' else 0)

# country cats
new_user_data['is_russia'] = new_user_data['country'].apply(lambda x: 1 if x == 'Russia' else 0)
# Расчет среднего exp_group для каждой страны
country_mean_age = new_user_data.groupby('country')['exp_group'].mean().to_dict()
new_user_data['city_mean_exp_group'] = new_user_data['country'].map(country_mean_age)

# генерация полиномов
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(new_user_data[['age', 'city_mean_age', 'city_median_age']])
interaction_df = pd.DataFrame(interaction_features, columns=poly.get_feature_names_out(['age', 'city_mean_age', 'city_median_age']))
new_user_data = pd.concat([new_user_data, interaction_df.drop(['age', 'city_mean_age', 'city_median_age'], axis=1)], axis=1)

new_user_data = new_user_data.rename(columns={
    'age city_mean_age': 'age_city_mean_age',
    'age city_median_age': 'age_city_median_age',
    'city_mean_age city_median_age': 'city_mean_age_city_median_age'
})

new_user_data.to_sql('ilja_pronin_gfr6897_lesson22_step_8_user_data_1', con=engine, if_exists='replace', index=False)