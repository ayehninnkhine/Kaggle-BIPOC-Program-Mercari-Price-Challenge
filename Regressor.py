import numpy as np
import pandas as pd
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("/Volumes/Data/Learning_Resources/Kaggle/mercari-train.tsv", sep="\t")
df['brand_name'].fillna('Not Available', inplace=True)
df['category_name'].fillna('Not Available', inplace=True)
df['item_description'].fillna('Not Available', inplace=True)

X = df[['item_condition_id', 'shipping']]
Y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# clf = RandomForestRegressor(n_jobs=-1,min_samples_leaf=3,n_estimators=200)
# clf = DecisionTreeRegressor(random_state=0)
clf = LGBMRegressor()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
rmsle = np.sqrt(mean_squared_log_error(y_test, pred))
print(rmsle)
