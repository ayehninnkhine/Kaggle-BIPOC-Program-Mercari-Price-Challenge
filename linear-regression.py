import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("/Volumes/Data/Learning_Resources/Kaggle/mercari-train.tsv", sep="\t")
df['brand_name'].fillna('Not Available', inplace=True)
df['category_name'].fillna('Not Available', inplace=True)
df['item_description'].fillna('Not Available', inplace=True)

X = df[['item_condition_id', 'shipping']]
Y = df['price']
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3)


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
