import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from collections import Counter

# Load data
df_games = pd.read_csv('games.csv', usecols=['app_id', 'is_free'], dtype={'app_id': str}, low_memory=False)
df_tags = pd.read_csv('tags.csv', usecols=['app_id', 'tag'], dtype={'app_id': str}, low_memory=False)
df_reviews = pd.read_csv('reviews.csv', usecols=['app_id', 'review_score'], dtype={'app_id': str}, low_memory=False)
df_genres = pd.read_csv('genres.csv', usecols=['app_id', 'genre'], dtype={'app_id': str}, low_memory=False)
df_categories = pd.read_csv('categories.csv', usecols=['app_id', 'category'], dtype={'app_id': str}, low_memory=False)

# join categories with spaces
df_categories['category'] = df_categories['category'].fillna("").apply(lambda x: x.replace(" ", "_"))
df_tags['tag'] = df_tags['tag'].fillna("").apply(lambda x: x.replace(" ", "_"))
df_genres['genre'] = df_genres['genre'].fillna("").apply(lambda x: x.replace(" ", "_"))

# group data
tags_grouped = df_tags.groupby('app_id')['tag'].apply(lambda x: ' '.join(x)).reset_index()
genres_grouped = df_genres.groupby('app_id')['genre'].apply(lambda x: ' '.join(x)).reset_index()
categories_grouped = df_categories.groupby('app_id')['category'].apply(lambda x: ' '.join(x)).reset_index()

# merge dataframes
df = df_games.merge(tags_grouped, on='app_id', how='left')
df = df.merge(genres_grouped, on='app_id', how='left')
df = df.merge(categories_grouped, on='app_id', how='left')
df = df.merge(df_reviews, on='app_id', how='left')

df.rename(columns={
    'tag': 'tags_str',
    'genre': 'genres_str',
    'category': 'categories_str'
}, inplace=True)

df['tags_str'] = df['tags_str'].fillna('')
df['genres_str'] = df['genres_str'].fillna('')
df['categories_str'] = df['categories_str'].fillna('')

df['review_score'] = pd.to_numeric(df['review_score'], errors='coerce')
df = df.dropna(subset=['review_score'])

# remove review score 0
df = df[df['review_score'] != 0]

# ensure that 'is_free' is numeric
if df['is_free'].dtype == 'object':
    df['is_free'] = df['is_free'].map({'True': 1, 'False': 0})
else:
    df['is_free'] = df['is_free'].astype(int)

# vectorize
vectorizer_tags = CountVectorizer()
tags_features = vectorizer_tags.fit_transform(df['tags_str'])

vectorizer_genres = CountVectorizer()
genres_features = vectorizer_genres.fit_transform(df['genres_str'])

vectorizer_categories = CountVectorizer()
categories_features = vectorizer_categories.fit_transform(df['categories_str'])

is_free_sparse = csr_matrix(df[['is_free']].values)

# combine all
X_sparse = hstack([tags_features, genres_features, categories_features])
X = hstack([X_sparse, is_free_sparse])

# add prefixes
features_tags = ["tag_" + name for name in vectorizer_tags.get_feature_names_out()]
features_genres = ["genre_" + name for name in vectorizer_genres.get_feature_names_out()]
features_categories = ["cat_" + name for name in vectorizer_categories.get_feature_names_out()]
features_is_free = ["is_free"]
all_feature_names = features_tags + features_genres + features_categories + features_is_free

# define review score as target
y = df['review_score'].values.astype(np.float32)

# split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

# prepare dataset
lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=all_feature_names)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train, feature_name=all_feature_names)

# set params
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 128,
    'max_depth': 12,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_gain_to_split': 0.01,
    'verbose': -1,
    'seed': 42
}

evals_result = {}

# early stopping and log
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=10),
    lgb.record_evaluation(evals_result)
]

# train
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train', 'eval'],
    callbacks=callbacks
)

# predict
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R^2:", r2)

# ______graphs and data_______


ax = lgb.plot_importance(model, max_num_features=20, importance_type='gain', figsize=(10, 8))
plt.title("Top 20 Feature Importances (Gain)")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

train_rmse = evals_result['train']['rmse']
val_rmse = evals_result['eval']['rmse']
plt.figure(figsize=(10, 6))
plt.plot(train_rmse, label='Training RMSE')
plt.plot(val_rmse, label='Validation RMSE')
plt.xlabel('Boosting Rounds')
plt.ylabel('RMSE')
plt.title('Training and Validation RMSE over Boosting Rounds')
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # 45° reference line
plt.xlabel('Actual Review Score')
plt.ylabel('Predicted Review Score')
plt.title('Actual vs. Predicted Review Score')
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.75)
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot([y_test, y_pred], labels=['Actual', 'Predicted'])
plt.title('Boxplot of Actual vs. Predicted Review Scores')
plt.show()

corr_coef = np.corrcoef(y_test, y_pred)[0, 1]
print("Correlation coefficient between actual and predicted review scores:", corr_coef)

df_test = df.loc[idx_test].copy()
df_test['predicted_review_score'] = y_pred

top10 = df_test.sort_values(by='predicted_review_score', ascending=False).head(10)
print("Top 10 gier o najwyższej przewidywanej ocenie recenzji:")
print(top10[['app_id', 'tags_str', 'genres_str', 'categories_str', 'review_score', 'predicted_review_score']])

top_threshold = df_test['predicted_review_score'].quantile(0.9)
bottom_threshold = df_test['predicted_review_score'].quantile(0.1)

top_decile = df_test[df_test['predicted_review_score'] >= top_threshold]
bottom_decile = df_test[df_test['predicted_review_score'] <= bottom_threshold]


def count_tokens(series):
    tokens = []
    for s in series:
        tokens.extend(s.split())
    return Counter(tokens)


top_tags = count_tokens(top_decile['tags_str'])
bottom_tags = count_tokens(bottom_decile['tags_str'])

top_tags_df = pd.DataFrame(top_tags.items(), columns=['tag', 'count']).sort_values(by='count', ascending=False).head(10)
bottom_tags_df = pd.DataFrame(bottom_tags.items(), columns=['tag', 'count']).sort_values(by='count',
                                                                                         ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(top_tags_df['tag'], top_tags_df['count'], color='blue')
plt.title('Top 10 tagów w grach o wysokiej przewidywanej ocenie')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
plt.bar(bottom_tags_df['tag'], bottom_tags_df['count'], color='red')
plt.title('Top 10 tagów w grach o niskiej przewidywanej ocenie')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

X_test_dense = X_test.toarray()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_dense)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
elif shap_values.ndim == 3 and shap_values.shape[0] == 1:
    shap_values = shap_values[0]
shap.summary_plot(shap_values, X_test_dense, feature_names=all_feature_names)
