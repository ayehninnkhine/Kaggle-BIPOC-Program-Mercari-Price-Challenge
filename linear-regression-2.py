import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("/Volumes/Data/Learning_Resources/Kaggle/mercari-train.tsv", sep="\t")


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


df['main_category'], df['subcat_1'], df['subcat_2'] = zip(*df['category_name'].apply(lambda x: split_cat(x)))


def toNumeric(data, to):
    if df[data].dtype == type(object):
        le = preprocessing.LabelEncoder()
        df[to] = le.fit_transform(df[data].astype(str))


toNumeric('name', 'n_name')
toNumeric('category_name', 'n_category_name')
toNumeric('brand_name', 'n_brand_name')
toNumeric('main_category', 'n_main_category')
toNumeric('subcat_1', 'n_subcat_1')
toNumeric('subcat_2', 'n_subcat_2')


# print(df.isnull().any())

def fill_missing_data(data):
    data.category_name.fillna(value="Other/Other/Other", inplace=True)
    data.brand_name.fillna(value="Unknown brand", inplace=True)
    data.item_description.fillna(value="No description", inplace=True)
    return data


df = fill_missing_data(df)

r_data = df[
    ['item_condition_id', 'shipping', 'n_name', 'n_brand_name', 'n_main_category', 'n_subcat_1', 'n_subcat_2']]
X_train, x_test, Y_train, y_test = train_test_split(r_data, df['price'], test_size=0.3)


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
model_1, rmsle_1 = run_model(lr, X_train, Y_train, x_test, y_test)

rf = RandomForestRegressor(n_jobs=-1, min_samples_leaf=3, n_estimators=200, max_features='sqrt', max_depth=30)
print("Random Forest Regression")
print("----------------")
model_2, rmsle_2 = run_model(rf, X_train, Y_train, x_test, y_test)

dt = DecisionTreeRegressor(random_state=0)
print("Decision Tree Regression")
print("----------------")
model_3, rmsle_3 = run_model(dt, X_train, Y_train, x_test, y_test)

gb = GradientBoostingRegressor()
print("Gradient Boosting Regression")
print("----------------")
model_4, rmsle_4 = run_model(gb, X_train, Y_train, x_test, y_test)