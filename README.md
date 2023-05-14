# House-Sales-in-King-County-USA

Display the data types of each column using the function dtypes, then take a screenshot and submit it, include your code in the image.

df.dtypes
---------------------------------------------------------------------------------------

Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() to obtain a statistical summary of the data. Take a screenshot and submit it, make sure the inplace parameter is set to True

df.drop("id", axis = 1, inplace = True)
df.drop("Unnamed: 0", axis = 1, inplace = True)
​
df.describe()
-----------------------------------------------------------------------------------------------------------------

Use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a dataframe.

df['floors'].value_counts().to_frame()

-------------------------------------------------------------------------------------------------------------


Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers.

sns.boxplot(x="waterfront", y="price", data=df)



-----------------------------------------------------------------------------------------------------


Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price.

sns.regplot(x="sqft_above", y="price", data=df, ci = None)

--------------------------------------------------------------------------------------------------

Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2. Take a screenshot of your code and the value of the R^2.

X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)

-------------------------------------------------------------------------------------------------

Fit a linear regression model to predict the 'price' using the list of features:

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     
Then calculate the R^2. Take a screenshot of your code.

X = df[features]
Y= df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)

----------------------------------------------------------------------------

Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list features, and calculate the R^2.

pipe=Pipeline(Input)
pipe
Pipeline(steps=[('scale', StandardScaler()),
                ('polynomial', PolynomialFeatures(include_bias=False)),
                ('model', LinearRegression())])
pipe.fit(X,Y)
Pipeline(steps=[('scale', StandardScaler()),
                ('polynomial', PolynomialFeatures(include_bias=False)),
                ('model', LinearRegression())])
pipe.score(X,Y)

-----------------------------------------------------------------------------------

Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.

from sklearn.linear_model import Ridge
RidgeModel = Ridge(alpha = 0.1)
RidgeModel.fit(x_train, y_train)
RidgeModel.score(x_test, y_test)

----------------------------------------------------------------------------------------------

Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2.

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
poly = Ridge(alpha=0.1)
poly.fit(x_train_pr, y_train)
poly.score(x_test_pr, y_test)
