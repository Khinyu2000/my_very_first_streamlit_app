import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
pd. set_option("display.max_columns", None)
dataset_file_path = './train.csv'
data = pd.read_csv(dataset_file_path)

y = data.SalePrice
all_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'TotRmsAbvGrd', 'SalePrice']
full_dataset = data[all_features]
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'TotRmsAbvGrd']
X = data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = RandomForestRegressor(random_state=1)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
model_MAE = mean_absolute_error(val_y, val_predictions)
#
st.title("Try to Predict Your House's Sale Price !")
st.markdown("We trained a model to predict the sale price of the specific house with the help of Random Forest Algorithm. This is the dataset that is used to train the model from Kaggle's Housing Prices Competition.")
full_dataset
print(full_dataset.describe())
st.markdown("The Mean Absolute Error of this Regression model : {:0.2f}".format(model_MAE))
print(type(val_X))
# pre_price = model.predict(pd.DataFrame({
#     'LotArea': [9600],
#     'YearBuilt': [1976],
#     '1stFlrSF': [856],
#     '2ndFlrSF': [854],
#     'FullBath': [2],
#     'TotRmsAbvGrd': [6],
# }))
st.markdown(""
            "")
st.subheader("Now, enter your house's features and get the price...")
form = st.form(key='my_form')
lotarea = form.slider(label = 'Select house\'s lot area', min_value = 1300, max_value = 215245)
builtYear = form.selectbox('Select the built year', range(1872, 2011, 1), key=1)
onest_floor_area = form.slider(label = 'Select 1st floor area', min_value = 334, max_value = 4692)
secondrd_floor_area = form.slider(label = 'Select 2rd floor area', min_value = 0, max_value = 2065)
bathroom = form.selectbox('Select number of bathroom', [1, 2, 3, 4, 5])
total_room_above_grade = form.slider(label= 'Select total rooms', min_value = 6, max_value = 17)
submit = form.form_submit_button('Get the sale price')

his_house = {
    'LotArea': [lotarea],
    'YearBuilt': [builtYear],
    '1stFlrSF': [onest_floor_area],
    '2ndFlrSF': [secondrd_floor_area],
    'FullBath': [bathroom],
    'TotRmsAbvGrd': [total_room_above_grade],
}


if submit:
    st.markdown(f"Your house is worth $ {str(model.predict(pd.DataFrame(his_house))[0])}")



