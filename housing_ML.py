#Example
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_name = 'melb_data.csv'

melbourne_housing_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "dansbecker/melbourne-housing-snapshot",
    file_name,
)

melbourne_housing_df.describe()
print(melbourne_housing_df.columns)

#drop na
melbourne_housing_data = melbourne_housing_df.dropna(axis=0)

#select the prediction target (dot notation) - pull out a variable
y = melbourne_housing_data.Price

#the features - factors that affect the y (x)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_housing_data[melbourne_features]
x.describe()
x.head()

#scikit-learn used to create models
from sklearn.tree import DecisionTreeRegressor #pip install scikit-learn

# Define model. Specify a number for random_state to ensure same results each run
# random_state = 1, ensures you get same results in each run. However, ML models allow randomness in model training
melbourne_model = DecisionTreeRegressor(random_state=1)  
# Fit model
melbourne_model.fit(x, y)

print("Making predictions for the following 5 houses:")
print("The predictions are")
print(melbourne_model.predict(x.head()))

#How good is the MODEL
#Mean Absolute Error (MAE)

from sklearn.metrics import mean_absolute_error
#prediction error
predicted_home_prices = melbourne_model.predict(x)
print(mean_absolute_error(y, predicted_home_prices))

# trainig data vs validation data 
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target The split is based on a random number generator. 
# Supplying a numeric value to the random_state argument guarantees we get the same split every time we run this script.
# split data into training and validation data, for both features and target
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_x, train_y)
# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions))

#compare MAE scores from different values for max_leaf_nodes
# max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    melbourne_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    melbourne_model.fit(train_x, train_y)
    preds_val=melbourne_model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# We can use a for-loop to compare the accuracy of models built with different values for max_leaf_nodes.
for max_leaf_nodes in [5, 50, 500, 5000]:
    compare_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print(f"Max leaf nodes: {max_leaf_nodes}  \t\t Mean Absolute Error:  {compare_mae: ,.0f}")