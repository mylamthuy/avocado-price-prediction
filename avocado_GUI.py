import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
from math import sqrt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from fbprophet import Prophet 
from fbprophet.plot import add_changepoints_to_plot
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv("avocado.csv")

#--------------
# GUI
st.title("Data Science Project")
st.write("## Hass Avocado Price Prediction")

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.to_csv("avocado_new.csv", index = False)
data.drop('Unnamed: 0', axis=1, inplace=True)

# Load regression model
with open('models/price_regression.pkl', 'rb') as pkl:
    reg_model = pickle.load(pkl)
# Load Arima model
with open('models/cali_organic_arima.pkl', 'rb') as pkl:
    arima_model = pickle.load(pkl)
# Load prophet model
with open('models/cali_conventional_fbprophet.pkl', 'rb') as pkl:
    pro_model2 = pickle.load(pkl)
# Load model
with open('models/chi_organic_arima.pkl', 'rb') as pkl:
    arima_model1 = pickle.load(pkl)
# Load model
with open('models/dfw_conventional_arima.pkl', 'rb') as pkl:
    arima_model2 = pickle.load(pkl)


# -----------------------------------
## GUI
menu = ["Business Objective", "Part 1", "Part 2", "Part 3", "Part 4"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### “Hass” Avocado, a company based in Mexico, specializes in the production of a variety of avocados sold in the US. They have been very successful in recent years and want to expand. Therefore, they wanted to build a reasonable model to predict the average price of “Hass” avocados in the US in order to consider the expansion of existing avocado farm types for avocado growing in other regions.
    - Part 1: Build Model for USA's Avocado Average Price Prediction
    - Part 2: Organic Avocado Average Price Prediction for the future in California 
    - Part 3: Conventioncal Avocado Average Price Prediction for the future in California
    - Part 4: Region Selection for the Business Expansion""")
    st.image("hass_avo.jpg")

elif choice == 'Part 1':
    st.subheader("Part 1: Build Model for USA’s Avocado Average Price Prediction - Regression Algorithm")
    ## PREDICT AVERAGEPRICE WITH REGRESSION ALGORITHM
    # Make new dataframe to use
    df = data[data['region']!='TotalUS']
    # Preprocessing
    def to_season(month):
        if month == 3 or month == 4 or month == 5:
            return 0
        elif month == 6 or month == 7 or month == 8:
            return 1
        elif month == 9 or month == 10 or month == 11:
            return 2
        else:
            return 3
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Month'].apply(lambda x: to_season(x))

        # Scaling Data
    df['TotalVolume_log'] = np.log1p(df['Total Volume'])
    df['TotalBags_log'] = np.log1p(df['Total Bags'])

        # Feature engineering
    le = LabelEncoder()
    df['type_le'] = le.fit_transform(df['type'])
    dfOneHot = pd.get_dummies(df, columns=['region'], drop_first=True)

    # Modeling
    X = dfOneHot.drop(['Date', 'AveragePrice','Total Volume', '4046', '4225', '4770',
        'Total Bags','Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'Season'], axis=1)
    y = df['AveragePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state =42)

        # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

        # Prediction & Evaluation
    y_pred = reg_model.predict(X_test_scaled)
            # Evaluation
    train_score = reg_model.score(X_train_scaled, y_train)
    test_score = reg_model.score(X_test_scaled, y_test)

    kf = KFold(n_splits=5)
    result_etr_train = cross_val_score(reg_model, X_train_scaled, y_train, cv=kf)
    train_mean = result_etr_train.mean()
    train_std = result_etr_train.std()

    result_etr_test = cross_val_score(reg_model, X_test_scaled, y_test, cv=kf)
    test_mean = result_etr_test.mean()
    test_std = result_etr_test.std()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mse)


    ## GUI
    st.write("""##### Some data:
    """)
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))
    st.write("##### General statistics:")
    st.dataframe(df.describe())
    st.write("#### Model Evaluation")
    st.code('Train accuracy score: {}'.format(round(train_score,4)))
    st.code('Test accuracy score: {}'.format(round(test_score,4)))
    st.write("The result shows that the model is good to predict avocado average price in the US with high accuracy (Test set shows 92% accuracy")
    st.write("Evaluating model with Cross Validation also shows high accuracy score and a stability with a small standard deviation:")
    st.write("Train accuracy score: {} +- {}".format(round(train_mean,4),round(train_std,4)))
    st.write("Test accuracy score: {} +- {}".format(round(test_mean,4),round(test_std,4)))
    st.code('MSE: {}'.format(round(mse,4)))
    st.code('MAE: {}'.format(round(mae,4)))
    st.code('RMSE: {}'.format(round(rmse,4)))
    st.write("##### Result Visualization")
    # Visualization
    figa, ax1 = plt.subplots()
    ax1=sns.distplot(y_train, hist=False, color="green", label="Actual Train Values")
    ax1=sns.distplot(reg_model.predict(X_train_scaled), hist=False, color="orange", label="Predicted Train Values")
    plt.legend()
    st.pyplot(figa)
    
    figb, ax2 = plt.subplots()
    ax2 = sns.distplot(y_test, hist=False, color="green", label="Actual Test Values")
    ax2 = sns.distplot(y_pred, hist=False, color="orange", label="Predicted Test Values")
    plt.legend()
    st.pyplot(figb)

    ## Download
    # Make new prediction and download result
    new_data = data[data['region']!='TotalUS'].reset_index(drop=True)
    scaled_data = scaler.transform(X)
    prediction = reg_model.predict(scaled_data)
    new_data['Prediction'] = prediction

    st.download_button(
    label="Download prediction data as CSV",
    data=new_data.to_csv(),
    file_name='prediction_data.csv',
    mime='text/csv')

elif choice == 'Part 2':
    st.subheader("Part 2: Organic Avocado Average Price Prediction for the future in California")

    ## ORGANIC AVOCADO Average Price Prediction In California
    # Make new dataframe from original dataframe: data
    df_ca = data[data['region'] == 'California']
    df_ca['Date'] = df_ca['Date'].str[:-3]
    df_ca = df_ca[df_ca['type'] == 'organic']

    agg = {'AveragePrice': 'mean'}
    df_ca_gr = df_ca.groupby(df_ca['Date']).aggregate(agg).reset_index()

    df_tso = pd.DataFrame()
    df_tso['ds'] = pd.to_datetime(df_ca_gr['Date'])
    df_tso['y'] = df_ca_gr['AveragePrice']

    ## ARIMA Algorithm
    df_ts1 = df_tso.copy(deep=False)
    df_ts1.index = pd.to_datetime(df_ts1.ds)
    df_ts1 = df_ts1.drop(['ds'], axis=1)
    # Decomposition
    decompose = seasonal_decompose(df_ts1.y, model='multiplicative')
    
        ## Split data into train and test
    train = df_ts1.loc['2015-01-01':'2017-05-01']
    test = df_ts1.loc['2017-06-01':]
        ## Forecast test set
    test_forecast = arima_model.predict(n_periods=len(test))
    test_forecast_table = pd.DataFrame(test_forecast, index=test.index, columns=['Test Forecast'])
    # Evaluation
    rmse = sqrt(mean_squared_error(test, test_forecast))
    mae = mean_squared_error(test, test_forecast)
    # Prediction for the next 12 months
    future_forecast_12 = arima_model.predict(n_periods=len(test)+12)
    future_forecast_table = pd.DataFrame(future_forecast_12, index=future_forecast_12.index[-12:], columns=['Future_forecast'])
    # Prediction for the next 5 years and visualization
    future_forecast_5 = arima_model.predict(n_periods=len(test)+12*5)
    future_forecast_table_5 = pd.DataFrame(future_forecast_5, index=future_forecast_5.index[-60:], columns=['Future_forecast'])

    ##GUI
    st.write("""
    ##### Some data:
    """)
    st.dataframe(df_tso.head(3))
    st.dataframe(df_tso.tail(3))
    st.text("Mean of Organic Avocado AveragePrice in California: " + str(round(df_tso['y'].mean(),2)) + " USD")
    st.text("Standard deviation of Organic Avocado AveragePrice in California: " + str(round(df_tso['y'].std(),2)) + " USD")

    ###### ARIMA ALGORITHM ######
    st.write("### Arima Algorithm")
    # Avocado Average Price in California
    fig1, ax = plt.subplots()
    ax.plot(df_ts1.index, df_ts1.y)
    ax.set_title("Organic Avocado Average Price in California", {'fontsize':12})
    plt.xticks(rotation=60)
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price')
    st.pyplot(fig1)
    # Visualize the result
    # Trend
    fig2, ax = plt.subplots()    
    ax.plot(decompose.trend)
    ax.set_title('Trend Pattern of Organic Avocado Average Price in California')
    plt.xticks(rotation=60)
    st.pyplot(fig2)
    # Seasonal
    fig3, ax = plt.subplots() 
    ax.plot(decompose.seasonal)
    ax.set_title('Seasonal Pattern of Organic Avocado Average Price in California')
    plt.xticks(rotation=60)
    st.pyplot(fig3)
    # Resid
    fig4, ax = plt.subplots() 
    ax.plot(decompose.resid)
    ax.set_title('Resid Pattern of Organic Avocado Average Pricein California')
    plt.xticks(rotation=60)
    st.pyplot(fig4)
    st.write("#### Modeling and Evaluation")
    code1 = '''arima_model = auto_arima(df_ts1,
                   start_p=2,
                   start_q=2,
                   max_p=3,
                   max_q=3,
                   m=12,
                   start_P=1,
                   start_Q=1,
                   d=1, D=1,
                   error_action='ignore',
                   trace=True,
                   suppress_warnings=True,
                   stepwise=True)'''
    st.code(code1,language='python')
    st.write("Train set tail:")
    st.dataframe(train.tail())
    st.write("Test set head:")
    st.dataframe(test.head())
    # Evaluation
    st.write("#### Evaluation:")
    st.code('RMSE = {}'.format(round(rmse,4)))
    st.code('MAE = {}'.format(round(mae,4)))
    st.text("This result shows that Arima model is good to predict the Average price of Organic avocado in California:")
    st.text("- RMSE is smaller than standard deviation of organic avocado data (0.27)")
    st.text("- MAE is 0.95% of mean AveragePrice of organic avocado data (1.68)")
    # Visualization
    st.write("#### Visualizing prediction for test set, the next 12 months and 5 years")
    fig5, ax = plt.subplots()
    ax.plot(df_ts1, label='Actual')
    ax.plot(test_forecast_table.index, test_forecast_table['Test Forecast'], label='Test Forecast')
    ax.plot(future_forecast_table.index, future_forecast_table['Future_forecast'], label='Future Forecast 12 months')
    ax.plot(future_forecast_table_5.index, future_forecast_table_5['Future_forecast'], label='Future Forecast 5 years')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price')
    ax.legend()
    st.pyplot(fig5)
    st.markdown("\n**Based on the above result, it is possible to expand the cultivation/production and trading Organic avocados in California**")

elif choice == 'Part 3':
    st.subheader("Part 3: Conventional Avocado Average Price Prediction for the future in California")

    ## CONVENTIONAL AVOCADO Average Price Prediction In California
    # Make new dataframe from original dataframe: data
    df_ca2 = data[data['region'] == 'California']
    df_ca2['Date'] = df_ca2['Date'].str[:-3]
    df_ca2 = df_ca2[df_ca2['type'] == 'conventional']

    agg = {'AveragePrice': 'mean'}
    df_ca_gr2 = df_ca2.groupby(df_ca2['Date']).aggregate(agg).reset_index()
    df_ca_gr2.head()

    df_tsc = pd.DataFrame()
    df_tsc['ds'] = pd.to_datetime(df_ca_gr2['Date'])
    df_tsc['y'] = df_ca_gr2['AveragePrice']

    ###### FACEBOOK PROPHET ALGORITHM ######
    df_ts4 = df_tsc.copy(deep=False)
    train = df_ts4.drop(df_ts4.index[-12:])
    test = df_ts4.drop(df_ts4.index[0:-12])

    # 12 months in test and 12 months to predict new values
    months = pd.date_range('2017-04-01','2019-03-01', freq='MS').strftime("%Y-%m-%d").tolist()    
    future = pd.DataFrame(months)
    future.columns = ['ds']
    future['ds'] = pd.to_datetime(future['ds'])
    # Use the model to make a forecast
    forecast = pro_model2.predict(future)
    y_test = test['y'].values
    y_pred = forecast['yhat'].values[:12]
    y_pred_12 = forecast['yhat'].tail(12).values
    rmse_p = sqrt(mean_squared_error(y_test, y_pred))
    mae_p = mean_absolute_error(y_test, y_pred)
    y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test['ds']),columns=['Actual'])
    y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test['ds']),columns=['Prediction Test'])
    y_pred_value_12 = pd.DataFrame(y_pred_12, index = pd.to_datetime(forecast['ds'][-12:]),columns=['Prediction 12 months'])

    # Long-term prediciton for the next 5 years
    m2 = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False) 
    m2.fit(df_ts4)
    future_new = m2.make_future_dataframe(periods=12*5, freq='M')
    forecast_new = m2.predict(future_new)

    ## GUI
    st.write("""
    ##### Some data:
    """)
    st.dataframe(df_tsc.head(3))
    st.dataframe(df_tsc.tail(3))
    st.text("Mean of Conventional Avocado AveragePrice in California: " + str(round(df_tsc['y'].mean(),2)) + " USD")
    st.text("Standard deviation of Conventional Avocado AveragePrice in California: " + str(round(df_tsc['y'].std(),2)) + " USD")

    st.write("### Facebook Prophet Algorithm")
    st.write("#### Modeling and Evaluation")
    st.code("""pro_model = Prophet(yearly_seasonality=True,
                    daily_seasonality=False, weekly_seasonality=False)""", language='python')
    st.write("##### Evaluation:")
    st.code('RMSE: {}'.format(round(rmse_p,4)))
    st.code('MAE: {}'.format(round(mae_p,4)))
    st.write("This result shows that Prophet model is good to predict the Average price of Conventioncal avocado in California:")
    st.write("- RMSE (=0.212) is equivalent to standard deviation of conventional avocado data (0.21)")
    st.write("- MAE (=0.087) is nearly 15% mean of AveragePrice of conventional avocado data (1.11)")

    # Visualization
    st.write("##### Next 12 months prediction")
    fig12 = pro_model2.plot(forecast) 
    fig12.show()
    a = add_changepoints_to_plot(fig12.gca(), pro_model2, forecast)
    st.pyplot(fig12)

    fig13 = pro_model2.plot_components(forecast)
    st.pyplot(fig13)

    st.dataframe(forecast[["ds", "yhat"]].tail(12))
    fig14, ax = plt.subplots()
    ax.plot(y_test_value, label='AveragePrice')
    ax.plot(y_pred_value, label='AveragePrice Prediction for Test data')
    ax.plot(y_pred_value_12, label='AveragePrice Prediction for the next 12 months')
    plt.xticks(rotation=60)
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price')
    ax.legend()
    st.pyplot(fig14)

    st.write("#### Long-term prediciton for the next 5 years")
    fig15 = m2.plot(forecast_new)
    a2 = add_changepoints_to_plot(fig15.gca(), m2, forecast_new)
    st.pyplot(fig15)

    fig16 = m2.plot_components(forecast_new)
    st.pyplot(fig16)

    fig17, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_ts4['y'], label='AveragePrice')
    ax.plot(forecast_new['yhat'], label='AveragePrice with next 60 months prediction',color='red')
    ax.legend()
    st.pyplot(fig17)
    st.markdown("**Based on the above result, it is possible to expand the cultivation/production and trading Conventional avocados in California**")

elif choice == 'Part 4':
    st.subheader("Part 4: Business Expansion")

    removed_region = ['California','GreatLakes','Midsouth','NewYork','Northeast','NorthernNewEngland',
        'Plains','SouthCarolina', 'SouthCentral','Southeast','TotalUS','West','WestTexNewMexico',
        'LosAngeles','Sacramento','SanDiego','SanFrancisco']
    df = data[~data['region'].isin(removed_region)]
    df['month_year'] = df['Date'].str[:-3]
    df['Date'] = pd.to_datetime(df['Date'])
    df['revenue'] = df['AveragePrice']*df['Total Volume']
    df_or = df[df['type']=='organic']
    df_con = df[df['type']=='conventional']


    ## GUI
    # -----EDA-----
    st.markdown("There are some big regions and states (such as California, GreatLakes, Midsouth, NewYork, Northeast,...) included in the data that affects the analysis, therefore, I delete these data from the data.")
    st.code("Data shape after deleting: {} rows and {} columns".format(df.shape[0], df.shape[1]))
    st.write("##### Some data:")
    st.dataframe(df.head(3))
    st.text("Mean of Avocado AveragePrice: " + str(round(df['AveragePrice'].mean(),2)) + " USD")

### ORGANIC AVOCADO
    st.write("### ORGANIC AVOCADO")
    st.markdown("**Chicago has a potential for the business expansion of Organic avocado because it has high value in all features and it shows an upward trend of revenue over the period. This will be demonstated based on the following graph**")

    # Average price based on region
    price_or = df_or.groupby(['region'])['AveragePrice'].mean().reset_index().sort_values('AveragePrice')
    fig18, ax = plt.subplots()
    ax = sns.barplot(x=price_or['region'], y = price_or['AveragePrice'])
    ax.set_title("Average Price of Organic Avocado by region")
    plt.xlabel(None,fontdict = {'fontsize':9})
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig18)

    # Check 'Total Volume' and 'region'
    or_sales = df_or.groupby(['region'])['Total Volume'].sum().sort_values(ascending=False)[:10].reset_index()
    or_bag = df_or.groupby(['region'])['Total Bags'].sum().sort_values(ascending=False)[:10].reset_index()
    or_revenue = df_or.groupby(['region'])['revenue'].sum().sort_values(ascending=False)[:10].reset_index()

    fig19, ax = plt.subplots()
    ax = sns.barplot(x='region', y='Total Volume', data = or_sales)
    plt.title('Total Volume and region',fontdict = {'fontsize':11})
    plt.xlabel(None,fontdict = {'fontsize':9})
    plt.xticks(rotation=90)
    st.pyplot(fig19)

    fig20, ax = plt.subplots()
    ax = sns.barplot(x='region', y='Total Bags', data = or_bag)
    plt.title('Total Bags and region',fontdict = {'fontsize':11})
    plt.xlabel(None,fontdict = {'fontsize':9})
    plt.xticks(rotation=90)
    st.pyplot(fig20)

    fig21, ax = plt.subplots()
    ax = sns.barplot(x='region', y='revenue', data = or_revenue)
    plt.title('revenue and region',fontdict = {'fontsize':11})
    ax.set_xlabel(None,fontdict = {'fontsize':9})
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig21)

    st.text("Mean of Organic Avocado AveragePrice: " + str(round((df_or['AveragePrice']).mean(),2)) + " USD")
    st.text("Chicago's Mean of Organic Avocado Average Price: " + str(round(df_or[df_or['region']=='Chicago']['AveragePrice'].mean(),2))+ " USD")
    st.write("As can be seen in the graphs, Chicago had high value in Total Volume and revenue, and the mean of AveragePrice is also higher the mean of organic avocado data.")

    st.write("##### Pick 10 regions with high value to see the trend over the period:")
    lst_10 = ['Seattle', 'Chicago', 'Portland', 'Denver', 'BaltimoreWashington',
       'Houston', 'DallasFtWorth', 'Philadelphia', 'Boston', 'Detroit']
    df_sub = df_or[df_or['region'].isin(lst_10)]
    agg1 = {'revenue': 'sum'}
    revenue_year = df_sub.groupby(['year','region']).aggregate(agg1).reset_index()

    fig22, ax = plt.subplots()
    palette = sns.color_palette("bright")
    ax = sns.lineplot(x='year', y='revenue', hue='region', data = revenue_year[revenue_year['year']!=2018], ci=None, palette = 'Set2')
    plt.legend(loc='upper left', borderaxespad=0, prop={'size': 6})
    plt.xlim(2015,2017,1)
    st.pyplot(fig22)

#### Use Arima algorithm to prediction the average price in Chicago
    st.write("#### Organic Avocado Avarage Price Prediction and Business Expansion Potential for the future in Chicago")
    # Make new dataframe from original dataframe: data
    df_chi = data[data['region'] == 'Chicago']
    df_chi['Date'] = df_chi['Date'].str[:-3]
    df_chi = df_chi[df_chi['type'] == 'organic']

    agg = {'AveragePrice': 'mean'}
    df_chi_gr = df_chi.groupby(df_chi['Date']).aggregate(agg).reset_index()

    df_tso = pd.DataFrame()
    df_tso['ds'] = pd.to_datetime(df_chi_gr['Date'])
    df_tso['y'] = df_chi_gr['AveragePrice']

    ## ARIMA Algorithm
    df_ts1 = df_tso.copy(deep=False)
    df_ts1.index = pd.to_datetime(df_ts1.ds)
    df_ts1 = df_ts1.drop(['ds'], axis=1)
    # Decomposition
    decompose = seasonal_decompose(df_ts1.y, model='multiplicative')
        ## Split data into train and test
    train = df_ts1.loc['2015-01-01':'2017-03-01']
    test = df_ts1.loc['2017-04-01':]
        ## Forecast test set
    test_forecast = arima_model1.predict(n_periods=len(test))
    test_forecast_table = pd.DataFrame(test_forecast, index=test.index, columns=['Test Forecast'])
    # Evaluation
    rmse = sqrt(mean_squared_error(test, test_forecast))
    mae = mean_squared_error(test, test_forecast)
    # Prediction for the next 5 years and visualization
    predict_5 = arima_model1.predict(n_periods=12*5)
    predict_table_5 = pd.DataFrame(predict_5, index=predict_5.index[-60:], columns=['Future_forecast'])

    ##GUI
    st.write("""
    ##### Some data:
    """)
    st.dataframe(df_tso.head(3))
    st.dataframe(df_tso.tail(3))
    st.write("Mean of Organic Avocado AveragePrice in Chicago: " + str(round(df_tso['y'].mean(),2)) + " USD")
    st.write("Standard deviation of Organic Avocado AveragePrice in Chicago: " + str(round(df_tso['y'].std(),2)) + " USD")

    ###### ARIMA ALGORITHM ######
    st.write("##### Arima Algorithm")
    # Avocado Average Price in California
    fig23, ax = plt.subplots()
    ax.plot(df_ts1.index, df_ts1.y)
    ax.set_title("Organic Avocado Average Price in Chicago", {'fontsize':12})
    plt.xticks(rotation=60)
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price')
    st.pyplot(fig23)
    # Visualize the result
    # Trend
    fig24, ax = plt.subplots()    
    ax.plot(decompose.trend)
    ax.set_title('Trend Pattern of Organic Avocado Average Price in Chicago')
    plt.xticks(rotation=60)
    st.pyplot(fig24)
    # Seasonal
    fig25, ax = plt.subplots() 
    ax.plot(decompose.seasonal)
    ax.set_title('Seasonal Pattern of Organic Avocado Average Price in Chicago')
    plt.xticks(rotation=60)
    st.pyplot(fig25)
    # Resid
    fig26, ax = plt.subplots() 
    ax.plot(decompose.resid)
    ax.set_title('Resid Pattern of Organic Avocado Average Price in Chicago')
    plt.xticks(rotation=60)
    st.pyplot(fig26)
    st.markdown("**Organic avocado average price data has seasonal and upward trend pattern.**")
    st.markdown("**Modeling and Evaluation**")
    st.write("Train set tail:")
    st.dataframe(train.tail())
    st.write("Test set head:")
    st.dataframe(test.head())
    # Evaluation
    st.code('RMSE = {}'.format(round(rmse,4)))
    st.code('MAE = {}'.format(round(mae,4)))
    st.write("This result shows that Arima model is good to predict the Average price of Organic avocado in Chicago:")
    st.text("- RMSE (=0.16) is lower than standard deviation of organic avocado data (0.23)")
    st.text("- MAE (=0.03) is 1,6% of mean AveragePrice of organic avocado data (1.74)")
    # Visualization
    st.write("##### Visualizing prediction")
    fig27, ax = plt.subplots()
    ax.plot(df_ts1, label='Actual')
    ax.plot(predict_table_5.index, predict_table_5['Future_forecast'], label='Future Forecast 5 years')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price')
    ax.legend()
    st.pyplot(fig27)
    st.markdown("\n**Based on the above result, it is possible to expand the cultivation/production and trading Organic avocados in Chicago**")

### CONVENTIONAL AVOCADO
    st.write("### CONVENTIONAL AVOCADO")
    st.markdown("**DallasFtWorth has a potential for the business expansion of Conventional avocado because it has high value in all features and it shows an upward trend of revenue over the period. This will be demonstated based on the following graph**")

    # Average price based on region
    price_con = df_con.groupby(['region'])['AveragePrice'].mean().reset_index().sort_values('AveragePrice')
    fig28, ax = plt.subplots()
    ax = sns.barplot(x=price_or['region'], y = price_or['AveragePrice'])
    ax.set_title("Average Price of Conventional Avocado by region")
    plt.xlabel(None,fontdict = {'fontsize':9})
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig28)

    # Check 'Total Volume' and 'region'
    con_sales = df_con.groupby(['region'])['Total Volume'].sum().sort_values(ascending=False)[:10].reset_index()
    con_bag = df_con.groupby(['region'])['Total Bags'].sum().sort_values(ascending=False)[:10].reset_index()
    con_revenue = df_con.groupby(['region'])['revenue'].sum().sort_values(ascending=False)[:10].reset_index()

    fig29, ax = plt.subplots()
    ax = sns.barplot(x='region', y='Total Volume', data = con_sales)
    plt.title('Total Volume and region',fontdict = {'fontsize':11})
    plt.xlabel(None,fontdict = {'fontsize':9})
    plt.xticks(rotation=90)
    st.pyplot(fig29)

    fig30, ax = plt.subplots()
    ax = sns.barplot(x='region', y='Total Bags', data = con_bag)
    plt.title('Total Bags and region',fontdict = {'fontsize':11})
    plt.xlabel(None,fontdict = {'fontsize':9})
    plt.xticks(rotation=90)
    st.pyplot(fig30)

    fig31, ax = plt.subplots()
    ax = sns.barplot(x='region', y='revenue', data = con_revenue)
    plt.title('revenue and region',fontdict = {'fontsize':11})
    ax.set_xlabel(None,fontdict = {'fontsize':9})
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig31)

    st.text("Mean of Conventional Avocado AveragePrice: " + str(round((df_con['AveragePrice']).mean(),2)) + " USD")
    st.text("DallasFtWorth's Mean of Conventional Avocado Average Price: " + str(round(df_con[df_con['region']=='DallasFtWorth']['AveragePrice'].mean(),2))+ " USD")
    st.write("As can be seen in the graphs, DallasFtWorth had high value in Total Volume and revenue.")

    st.write("##### Pick 10 regions with high value to see the trend over the period:")
    lst10 = ['BaltimoreWashington', 'DallasFtWorth', 'Chicago', 'Houston', 'Denver',
       'PhoenixTucson', 'Boston', 'MiamiFtLauderdale', 'Seattle', 'Portland']
    df_sub1 = df_con[df_con['region'].isin(lst10)]
    agg1 = {'revenue': 'sum'}
    revenue_year1 = df_sub1.groupby(['year','region']).aggregate(agg1).reset_index()

    fig32, ax = plt.subplots()
    palette = sns.color_palette("bright")
    ax = sns.lineplot(x='year', y='revenue', hue='region', data = revenue_year1[revenue_year1['year']!=2018], ci=None, palette = 'Set2')
    plt.legend(loc='upper left', borderaxespad=0, prop={'size': 6})
    plt.xlim(2015,2017,1)
    st.pyplot(fig32)

#### Use Arima algorithm to prediction the average price in DallasFtWorth
    st.write("#### Conventional Avocado Avarage Price Prediction and Business Expansion Potential for the future in DallasFtWorth")
    # Make new dataframe from original dataframe: data
    df_dal = data[data['region'] == 'DallasFtWorth']
    df_dal['Date'] = df_dal['Date'].str[:-3]
    df_dal = df_dal[df_dal['type'] == 'conventional']

    agg = {'AveragePrice': 'mean'}
    df_dal_gr = df_dal.groupby(df_dal['Date']).aggregate(agg).reset_index()

    df_tsc = pd.DataFrame()
    df_tsc['ds'] = pd.to_datetime(df_dal_gr['Date'])
    df_tsc['y'] = df_dal_gr['AveragePrice']

    ## ARIMA Algorithm
    df_ts3 = df_tsc.copy(deep=False)
    df_ts3.index = pd.to_datetime(df_ts3.ds)
    df_ts3 = df_ts3.drop(['ds'], axis=1)
    # Decomposition
    decompose2 = seasonal_decompose(df_ts3.y, model='multiplicative')
        ## Split data into train and test
    train = df_ts3.loc['2015-01-01':'2017-03-01']
    test = df_ts3.loc['2017-04-01':]
        ## Forecast test set
    test_forecast = arima_model2.predict(n_periods=len(test))
    test_forecast_table = pd.DataFrame(test_forecast, index=test.index, columns=['Test Forecast'])
    # Evaluation
    rmse = sqrt(mean_squared_error(test, test_forecast))
    mae = mean_squared_error(test, test_forecast)
    # Prediction for the next 5 years and visualization
    predict_5 = arima_model2.predict(n_periods=12*5)
    predict_table_5 = pd.DataFrame(predict_5, index=predict_5.index[-60:], columns=['Future_forecast'])

    ##GUI
    st.write("""
    ##### Some data:
    """)
    st.dataframe(df_tsc.head(3))
    st.dataframe(df_tsc.tail(3))
    st.text("Mean of Conventional Avocado AveragePrice in DallasFtWorth: " + str(round(df_tsc['y'].mean(),2)) + " USD")
    st.text("Standard deviation of Conventional Avocado AveragePrice in DallasFtWorth: " + str(round(df_tsc['y'].std(),2)) + " USD")

    ###### ARIMA ALGORITHM ######
    st.write("##### Arima Algorithm")
    # Avocado Average Price in DallasFtWorth
    fig33, ax = plt.subplots()
    ax.plot(df_ts3.index, df_ts3.y)
    ax.set_title("Conventional Avocado Average Price in DallasFtWorth", {'fontsize':12})
    plt.xticks(rotation=60)
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price')
    st.pyplot(fig33)
    # Visualize the result
    # Trend
    fig34, ax = plt.subplots()    
    ax.plot(decompose2.trend)
    ax.set_title('Trend Pattern of Conventional Avocado Average Price in DallasFtWorth')
    plt.xticks(rotation=60)
    st.pyplot(fig34)
    # Seasonal
    fig35, ax = plt.subplots() 
    ax.plot(decompose2.seasonal)
    ax.set_title('Seasonal Pattern of Conventional Avocado Average Price in v')
    plt.xticks(rotation=60)
    st.pyplot(fig35)
    # Resid
    fig36, ax = plt.subplots() 
    ax.plot(decompose2.resid)
    ax.set_title('Resid Pattern of Conventional Avocado Average Price in DallasFtWorth')
    plt.xticks(rotation=60)
    st.pyplot(fig36)
    st.markdown("**Conventional avocado average price data has seasonal and upward trend pattern.**")
    st.markdown("**Modeling and Evaluation**")
    st.write("Train set tail:")
    st.dataframe(train.tail())
    st.write("Test set head:")
    st.dataframe(test.head())
    # Evaluation
    st.code('RMSE = {}'.format(round(rmse,4)))
    st.code('MAE = {}'.format(round(mae,4)))
    st.write("This result shows that Arima model is good to predict the Average price of Conventional avocado in DallasFtWorth:")
    st.text("- RMSE (=0.12) is higher than standard deviation of conventional avocado data (0.14)")
    st.text("- MAE (=0.015) is nearly 1.7% mean of AveragePrice of organic avocado data (0.85)")
    # Visualization
    st.write("##### Visualizing prediction")
    fig37, ax = plt.subplots()
    ax.plot(df_ts3, label='Actual')
    ax.plot(predict_table_5.index, predict_table_5['Future_forecast'], label='Future Forecast 5 years')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price')
    ax.legend()
    st.pyplot(fig37)
    st.markdown("\n**Based on the above result, it is possible to expand the cultivation/production and trading Conventional avocados in DallasFtWorth**")









    



