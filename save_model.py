from catboost import CatBoostClassifier
from load_features_to_train import load_features


catboost_model = CatBoostClassifier(verbose=0, learning_rate=0.02, task_type='GPU')
data = load_features(5*10**4)

categorical_columns = data.select_dtypes(include='object').columns.to_list()
data[categorical_columns] = data[categorical_columns].astype(str)

print(categorical_columns)

X_train = data.drop('target', axis=1)
y_train = data['target']

catboost_model.fit(X_train, y_train, cat_features=categorical_columns)

catboost_model.save_model('catboost_model_step_8', format="cbm")
