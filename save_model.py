from catboost import CatBoostClassifier
from load_features_to_train import load_features


data = load_features(7*10**4)

categorical_columns = data.select_dtypes(include='object').columns.to_list()
data[categorical_columns] = data[categorical_columns].astype(str)

print(categorical_columns)

X = data.drop('target', axis=1)
y = data['target']

catboost_model = CatBoostClassifier(
    verbose=0,
    task_type='GPU',
    random_state=42,
    auto_class_weights='Balanced',
    depth=8,
    iterations=200,
    l2_leaf_reg=1,
    learning_rate=0.02
)
catboost_model.fit(X, y, cat_features=categorical_columns)
catboost_model.save_model('catboost_model_step_5', format="cbm")
