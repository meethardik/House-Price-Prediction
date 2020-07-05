import numpy as np
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from sklearn.preprocessing import StandardScaler

sf = open(r"Pandas_exer_Ecomm_Purchase/Data/final_data.csv", "r")

prices = pd.read_csv(sf, sep=",")
sf.close()

data = pd.DataFrame(prices)

data.drop(
    ["info", "z_address", "bathrooms", "zestimate", "zipcode", "zpid"],
    axis=1,
    inplace=True,
)

data["datetime_series"] = pd.to_datetime(data["lastsolddate"])
data = data.set_index("datetime_series")

data.zindexvalue = data.zindexvalue.str.replace(",", "")
data.zindexvalue = pd.to_numeric(data.zindexvalue)

# Find co-relation matrix of lastsoldprice field with other field(s)
corr_matrix = data.corr()
print(corr_matrix["lastsoldprice"].sort_values(ascending=False))

# finishedsqft is more promising in relation to lastsoldprice, let's have scatter plot of it
data.plot(
    kind="scatter", x=["finishedsqft"], y="lastsoldprice", alpha=0.5,
)
plt.xlabel("Lastsoldprice")
plt.ylabel("Finishedsqft")
plt.title("Finishedsqft field closely related to Lastsoldprice")
plt.show()

# As each neighbourhood have different house price, we need price per sqft
data["price_per_sqft"] = data["lastsoldprice"] / data["finishedsqft"]

# we need to group by the neighbourhood along with Address field
groupby_neighbor = data.groupby("neighborhood").count()["address"]
# Calculate Average price per sqft
avg_price_per_sqft = data.groupby("neighborhood").mean()["price_per_sqft"]

cluster = pd.concat([groupby_neighbor, avg_price_per_sqft], axis=1)
# print(cluster.index)
cluster["neighborhood"] = cluster.index
cluster.columns = ["groupby_neighbor", "price_per_sqft", "neighborhood"]

# break the neighborhood into three 1) Low price 2) High Price Low Frequency 3) high Price high frequency
cluster1 = cluster[cluster.price_per_sqft < 756]

cluster_temp = cluster[cluster.price_per_sqft >= 756]
cluster2 = cluster_temp[cluster_temp.groupby_neighbor < 123]

cluster3 = cluster_temp[cluster_temp.groupby_neighbor >= 123]

def getGroupByNeighborhood(name):
    if name in cluster1.index:
        return "low_price"
    elif name in cluster2.index:
        return "High_Price_Low_Freq"
    else:
        return "High_Price_High_Freq"


data["group"] = data["neighborhood"].apply(getGroupByNeighborhood)

# We can now drop columns address, lastsolddate, latitude, longitude, neighborhood,
data.drop(
    ["address", "lastsolddate", "latitude", "longitude", "neighborhood"],
    axis=1,
    inplace=True,
)
# print(data.head())

""" Now we need to separate out Categorical data from Continous one. 
Cat = Group and usercode and rest others are Continous
So we shall create dummy variables for group and usercode"""

# We shall now craete X and y variables to work with them ahead

# these are variables on which target depends on
X = data[
    [
        "bedrooms",
        "finishedsqft",
        "totalrooms",
        "usecode",
        "yearbuilt",
        "zindexvalue",
        "group",
        "price_per_sqft",
    ]
]

# this is target
y = data["lastsoldprice"]

# Create a dummy var for Group
dummy_group = pd.get_dummies(data["group"])
X = pd.concat([X, dummy_group], axis=1)

# Create dummy var for usecode
dummy_usecode = pd.get_dummies(data["usecode"])
X = pd.concat([X, dummy_usecode], axis=1)

dropCategoricalFields = ["group", "usecode"]

# drop above fields
X.drop(dropCategoricalFields, inplace=True, axis=1)
finalArrayOfColumns = X.columns.values
print(finalArrayOfColumns)

"""y have only 1 value, and X have both, so we need to standardize the data
So we should use from sklearn.preprocessing import StandardScaler """

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Now we use train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# We now work with GradientBoostingRegressor to get more accurate values
regressor = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=50,
    learning_rate=0.05,
    max_features="sqrt",
    subsample=0.5,
    random_state=10,
)

regressor.fit(X_train, y_train)

errors = [
    mean_squared_error(y_test, y_pred) for y_pred in regressor.staged_predict(X_test)
]

best_n_estimators = np.argmin(errors)

best_regressor = GradientBoostingRegressor(
    max_depth=2, learning_rate=0.05, n_estimators=best_n_estimators
)

best_regressor.fit(X_train, y_train)
y_pred = best_regressor.predict(X_test[0 : len(X_test)])

print("Mean Absolute Error : %.4f" % mean_absolute_error(y_test, y_pred))
print("Mean Squared Error : %.4f" % mean_squared_error(y_test, y_pred))
print("Root mean Squared Error : %.4f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("Model Score : %.4f " % best_regressor.score(X_test, y_test))

importanceValue = best_regressor.feature_importances_

# print("Importance values : ", importanceValue)
feature_indexes_by_importance = importanceValue.argsort()
# print("Values indexed by Importance : ", feature_indexes_by_importance)

sorted_idx = np.argsort(importanceValue)
pos = np.arange(sorted_idx.shape[0]) + 2.5
fig, ax = plt.subplots(figsize=(10, 6))

plt.yticks(pos, np.array(finalArrayOfColumns)[sorted_idx])
plt.title("Importance value matrix for Continous variables")
plt.barh(pos, importanceValue[sorted_idx], align="center")
plt.show()

plt.scatter(y_pred, y_train.iloc[0 : len(y_pred)], color="purple")
plt.title("Prediction Vs Train model")
plt.xlabel("Predictions", fontsize=14)
plt.ylabel("Train data", fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(y_pred, y_test, color="green")
plt.title("Actual Vs Predicted House Price Index")
plt.xlabel("Predictions", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.grid(True)
plt.show()