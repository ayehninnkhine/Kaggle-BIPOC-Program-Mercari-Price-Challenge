import string
from string import punctuation
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("/Volumes/Data/Learning_Resources/Kaggle/mercari-train.tsv", sep="\t")

punctuation_symbols = []
for symbol in punctuation:
    punctuation_symbols.append((symbol, ''))


def remove_punctuation(sentence: str) -> str:
    return sentence.translate(str.maketrans('', '', string.punctuation))


def remove_digits(x):
    x = ''.join([i for i in x if not i.isdigit()])
    return x


stop = stopwords.words('english')


def remove_stop_words(x):
    x = ' '.join([i for i in x.lower().split(' ') if i not in stop])
    return x


def to_lower(x):
    return x.lower()


def transform_category_name(category_name):
    try:
        main, sub1, sub2 = category_name.split('/')
        return main, sub1, sub2
    except:
        return np.nan, np.nan, np.nan


def handle_missing_values(df):
    df['category_name'].fillna(value='missing', inplace=True)
    df['brand_name'].fillna(value='None', inplace=True)
    df['item_description'].fillna(value='None', inplace=True)


def to_categorical(df):
    df['brand_name'] = df['brand_name'].astype('category')
    df['category_name'] = df['category_name'].astype('category')
    df['item_condition_id'] = df['item_condition_id'].astype('category')


handle_missing_values(df)

r_data = df[
    ['item_condition_id', 'shipping', 'name', 'brand_name', 'category_name', 'item_description']]
X_train, x_test, Y_train, y_test = train_test_split(r_data, df['price'], test_size=0.3)

X_train.item_description = X_train.item_description.astype(str)
X_train['item_description'] = X_train['item_description'].apply(remove_digits)
X_train['item_description'] = X_train['item_description'].apply(remove_punctuation)
X_train['item_description'] = X_train['item_description'].apply(remove_stop_words)
X_train['item_description'] = X_train['item_description'].apply(to_lower)
X_train['name'] = X_train['name'].apply(remove_digits)
X_train['name'] = X_train['name'].apply(remove_punctuation)
X_train['name'] = X_train['name'].apply(remove_stop_words)
X_train['name'] = X_train['name'].apply(to_lower)

x_test.item_description = x_test.item_description.astype(str)
x_test['item_description'] = x_test['item_description'].apply(remove_digits)
x_test['item_description'] = x_test['item_description'].apply(remove_punctuation)
x_test['item_description'] = x_test['item_description'].apply(remove_stop_words)
x_test['item_description'] = x_test['item_description'].apply(to_lower)
x_test['name'] = x_test['name'].apply(remove_digits)
x_test['name'] = x_test['name'].apply(remove_punctuation)
x_test['name'] = x_test['name'].apply(remove_stop_words)
x_test['name'] = x_test['name'].apply(to_lower)

cv = CountVectorizer(min_df=10)
X_name_train = cv.fit_transform(X_train['name'])
X_name_test = cv.fit_transform(x_test['name'])

cv = CountVectorizer()
X_category_train = cv.fit_transform(X_train['category_name'])
X_category_test = cv.fit_transform(x_test['category_name'])

tv = TfidfVectorizer(max_features=55000, ngram_range=(1, 2), stop_words='english')
X_description_train = tv.fit_transform(X_train['item_description'])
X_description_test = tv.fit_transform(x_test['item_description'])

lb = LabelBinarizer(sparse_output=True)
X_brand_train = lb.fit_transform(X_train['brand_name'])
X_brand_test = lb.fit_transform(x_test['brand_name'])

X_dummies_train = csr_matrix(pd.get_dummies(X_train[['item_condition_id', 'shipping']], sparse=True).values)
sparse_merge_train = hstack(
    (X_dummies_train, X_description_train, X_brand_train, X_category_train, X_name_train)).tocsr()

X_dummies_test = csr_matrix(pd.get_dummies(x_test[['item_condition_id', 'shipping']], sparse=True).values)
sparse_merge_test = hstack((X_dummies_test, X_description_test, X_brand_test, X_category_test, X_name_test)).tocsr()


def run_model(model, X_train, Y_train, x_test, y_test, verbose=False):
    # Y_train = Y_train[:, np.newaxis].ravel()
    model.fit(X_train, Y_train)
    y_predict = model.predict(x_test)
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))
    print(rmsle)
    return model, rmsle


lr = LinearRegression()
print("Linear Regression")
print("----------------")
model_1, rmsle_1 = run_model(lr, sparse_merge_train, Y_train, sparse_merge_test, y_test)

rf = RandomForestRegressor(n_jobs=-1, min_samples_leaf=3, n_estimators=200)
print("Random Forest Regression")
print("----------------")
model_2, rmsle_2 = run_model(rf, sparse_merge_train, Y_train, sparse_merge_test, y_test)

dt = DecisionTreeRegressor(random_state=0)
print("Decision Tree Regression")
print("----------------")
model_3, rmsle_3 = run_model(dt, sparse_merge_train, Y_train, sparse_merge_test, y_test)
