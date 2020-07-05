# House-Price-Prediction
The code predicts House Price from the data set provided. The data set is based on address from US. The solution calculates Co relation matrix between the required field(s) based on target field. A scatter plot showcasing relation between two fields lastsoldprice and finishedsqft. A cluster been created based on LowPrice, HighPriceHighFrequency and HighPriceLowFrequency. GradientBoostingRegressor is chosen to get better results as we have multiple parameters to work with. I had calculated Mean_Square_Error. Based on the value of Mean_Square_Error, a new value of estimator is been calculated using numPy.argmin(Mean_Square_Error). This estimator is then passed to calculate GradientBoostingRegressor. Y-Prediction is calculated using the regressor object that we obtain. Few other values calculated are Mean Absolute Error, Mean Squared Error, Root mean Squared Error and finally Model Score. I have also calculated Feature importance from regressor. A bar chart is generated using the Importance Matrix. Scatter plot is generated for Prediction and Train Model. At last but not least, Scatter plot for Actual and Predicted House Price. 
