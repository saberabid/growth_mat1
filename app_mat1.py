import streamlit as st
from prophet import Prophet
import pandas as pd
import plotly.graph_objs as go
import pickle
import os




# Hardcoded user credentials
users = {
    'Riad.GAHGOUHI': 'password123',
    'Saber.ABID': 'password456'
}

# Create a login form
st.title('Sales Forecasting with Prophet')

# Login section
st.sidebar.header('Login')
username = st.sidebar.text_input('Username')
password = st.sidebar.text_input('Password', type='password')

if st.sidebar.button('Login'):
    if username in users and users[username] == password:
        st.sidebar.success(f'Welcome {username}!')
        # Load your dataset from an Excel file
        file_path = r'C:\Users\business.analyse\Desktop\Projet prediction\forcasts mat1\dataset_sales_mat1-Copie.xlsx'
        df = pd.read_excel(file_path, sheet_name='Top_20_labs_mat1')

        # Convert sales data to billions during import
        sales_columns = [2019, 2020, 2021, 2022, 2023, 2024]
        df_sales = df[['LABORATOIRE'] + sales_columns]
        df_sales[sales_columns] = df_sales[sales_columns] / 1e9  # Convert sales to billions

        # Aggregate by 'LABORATOIRE'
        df_aggregated = df_sales.groupby(['LABORATOIRE']).sum().reset_index()

        # Prepare data in the correct format for Prophet
        df_long = pd.melt(df_aggregated, id_vars=['LABORATOIRE'], value_vars=sales_columns,
                          var_name='Year', value_name='Sales')
        df_long = df_long.rename(columns={'Year': 'ds', 'Sales': 'y'})
        df_long['ds'] = pd.to_datetime(df_long['ds'].astype(str), format='%Y')

        # Path to save trained models
        model_path = r'C:\Users\business.analyse\Desktop\Projet prediction\forcasts mat1\trained_models'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        def train_and_save_model(laboratoire, df):
            df_labo = df[df['LABORATOIRE'] == laboratoire].copy()
            df_labo = df_labo[['ds', 'y']]

            model = Prophet(yearly_seasonality=True)
            model.fit(df_labo)

            # Save model
            with open(os.path.join(model_path, f'{laboratoire}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)

        def load_model(laboratoire):
            try:
                with open(os.path.join(model_path, f'{laboratoire}_model.pkl'), 'rb') as f:
                    model = pickle.load(f)
            except FileNotFoundError:
                train_and_save_model(laboratoire, df_long)
                with open(os.path.join(model_path, f'{laboratoire}_model.pkl'), 'rb') as f:
                    model = pickle.load(f)
            return model

        def forecast_sales(laboratoire, years):
            model = load_model(laboratoire)
            
            # Create future dataframe for yearly data
            future = model.make_future_dataframe(periods=years, freq='Y')
            forecast = model.predict(future)
            
            return forecast

        def update_forecast_for_all_labs(years_to_predict):
            df_ranked = df_aggregated.copy()
            all_forecasts = {}

            # Loop through each lab to calculate forecasts
            for labo in df_aggregated['LABORATOIRE'].unique():
                forecast = forecast_sales(labo, years_to_predict)
                all_forecasts[labo] = forecast
                
                # Assign the forecast for future years to columns
                forecast_latest = forecast[forecast['ds'].dt.year >= 2025]
                for year in range(2025, 2025 + years_to_predict):
                    forecast_year = forecast_latest[forecast_latest['ds'].dt.year == year]
                    if not forecast_year.empty:
                        column_name = f'Forecast {year}'
                        df_ranked.loc[df_ranked['LABORATOIRE'] == labo, column_name] = forecast_year['yhat'].iloc[0]
            
            # Sort by the last predicted year
            last_year = 2025 + years_to_predict - 2
            df_ranked = df_ranked.sort_values(by=f'Forecast {last_year}', ascending=False)
            
            # Add a ranking column based on the last forecasted year
            df_ranked['Rank'] = range(1, len(df_ranked) + 1)
            
            return df_ranked, all_forecasts

        # The rest of your code follows as it is, including displaying the tables and plots


        def calculate_growth_rate(inpha_forecast, yellow_forecast, years_to_predict):
        # Calculate the growth rate needed for INPHA-MEDIS to reach the yellow lab's forecast
            growth_rates = []
            for year in range(2025, 2025 + years_to_predict - 1):
            # Fetch sales forecasts
                inpha_sales = inpha_forecast[inpha_forecast['ds'].dt.year == year]['yhat']
                yellow_sales = yellow_forecast[yellow_forecast['ds'].dt.year == year]['yhat']
            
            # Check if sales data exists for both forecasts
                if not inpha_sales.empty and not yellow_sales.empty:
                    inpha_sales_value = inpha_sales.values[0]
                    yellow_sales_value = yellow_sales.values[0]
                    if inpha_sales_value < yellow_sales_value:
                        growth_rate = (yellow_sales_value / inpha_sales_value) - 1
                        growth_rates.append(growth_rate * 100)  # Convert to percentage
                    else:
                        growth_rates.append(0)  # No growth needed if INPHA-MEDIS already exceeds or meets the yellow lab's forecast
                else:
                    growth_rates.append(0)  # Handle missing data by appending 0 or other placeholder
        
            return growth_rates

        def average_growth_rate(growth_rates):
            # Compute average growth rate
            valid_rates = [rate for rate in growth_rates if rate > 0]
            if valid_rates:
                return sum(valid_rates) / len(valid_rates)
            else:
                return 0

        st.title('Sales Forecasting with Prophet')

    # Sidebar for user inputs
        st.sidebar.header('User Inputs')
        years_to_predict = st.sidebar.slider('Number of Years to Predict:', 1, 10, 5) + 1
        rank_class = st.sidebar.slider('Select Rank Class to Highlight:', 1, 20, 5)

    # Update forecasts for all labs
        df_ranked, all_forecasts = update_forecast_for_all_labs(years_to_predict)

    # Highlight the rows based on the selected rank class and specific laboratory
        def highlight_row(row, rank_class):
            color = ''
            if row['Rank'] == rank_class:
                color = 'background-color: yellow'
            if row['LABORATOIRE'] == 'INPHA-MEDIS':
                color = 'background-color: lightcoral'
            return [color] * len(row)

    # Apply highlight
        df_styled = df_ranked.style.apply(highlight_row, rank_class=rank_class, axis=1)

    # Display the table with forecasts for all labs, highlighting the selected rank
        st.subheader('Forecasted Sales for All Laboratories')
        st.write(f"Displaying forecast for all 20 laboratories, sorted by predicted sales for {2025 + years_to_predict - 2}.")
        st.dataframe(df_styled)

    # Select specific lab for detailed forecast
        selected_laboratoire = st.sidebar.selectbox('Select Laboratory for Detailed Forecast:', df_aggregated['LABORATOIRE'].unique())

        if selected_laboratoire:
            forecast = all_forecasts[selected_laboratoire]
        
        # Plot the forecast for the selected laboratory
            fig_forecast = go.Figure()

            fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                          mode='lines+markers', name='Forecast',
                                          line=dict(color='cyan', width=2),
                                          marker=dict(symbol='circle', size=8, color='cyan')))

            fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                          mode='markers+text', name='Forecast Points',
                                          text=[f"{y:,.2f}" for y in forecast['yhat']],
                                          textposition='top center',
                                          marker=dict(symbol='circle', size=10, color='cyan', line=dict(width=2, color='blue'))))
        
            fig_forecast.update_layout(
            title=f'Sales Forecast for {selected_laboratoire}',
            xaxis_title='Date',
            yaxis_title='Sales (in billions)',
            xaxis=dict(
                tickformat='%Y',
                tickvals=pd.date_range(start=forecast['ds'].min(), end=forecast['ds'].max(), freq='Y')
            ),
            yaxis_tickformat=',.2f',
            template='plotly_dark',
            width=1200,
            height=600
            )
        
            st.plotly_chart(fig_forecast)

    # Calculate and display growth rates for INPHA-MEDIS vs. Yellow Lab
        inpha_forecast = all_forecasts['INPHA-MEDIS']
        yellow_lab_name = df_ranked[df_ranked['Rank'] == rank_class]['LABORATOIRE'].values[0]
        yellow_forecast = all_forecasts[yellow_lab_name]

        growth_rates = calculate_growth_rate(inpha_forecast, yellow_forecast, years_to_predict)
        avg_growth_rate = average_growth_rate(growth_rates)

    # Plot the growth rates for INPHA-MEDIS
        fig_growth = go.Figure()

        years_range = list(range(2025, 2025 + years_to_predict - 1))
        fig_growth.add_trace(go.Scatter(x=years_range,
                                    y=growth_rates,
                                    mode='lines+markers',
                                    name='Growth Rate',
                                    line=dict(color='magenta', width=2),
                                    marker=dict(symbol='circle', size=8, color='magenta')))

    # Add annotations for growth rates
        annotations = [dict(
        x=year,
        y=growth_rates[idx],
        text=f"{growth_rates[idx]:.2f}%",
        showarrow=True,
        arrowhead=2
        ) for idx, year in enumerate(years_range)]

        fig_growth.update_layout(
            title=f'Growth Rate for INPHA-MEDIS to Reach {yellow_lab_name} Forecast',
            xaxis_title='Year',
            yaxis_title='Growth Rate (%)',
            xaxis=dict(
                tickvals=years_range,
                ticktext=[str(year) for year in years_range]
            ),
            yaxis_tickformat='.2f',
            annotations=annotations,
            template='plotly_dark',
            width=1200,
            height=600
            )

        st.plotly_chart(fig_growth)

    # Display average growth rate
        st.subheader('Average Growth Rate')
        st.write(f"The average growth rate needed for INPHA-MEDIS to achieve the same forecast as the {yellow_lab_name} is: {avg_growth_rate:.2f}%")
        
    else:
        st.sidebar.error('Username/password is incorrect')
else:
    st.stop()  # Stop execution if not logged in


