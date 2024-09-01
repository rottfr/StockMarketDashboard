import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from modules.utils import stock_options, fetch_dow_jones_tickers, fetch_sp500_tickers, fetch_nasdaq100_tickers, fetch_dow_jones_transportation_tickers, fetch_dow_jones_utility_tickers

def apply_performance_color(row, is_stock=False):
    """Apply text color to the row based on price performance."""
    color = 'color: '
    
    if is_stock:
        current_price = row['Current Price']
        starting_price = row['Starting Price']
    else:
        closing_price = float(row['Closing Price'].replace('$', '').replace(',', ''))
        opening_price = float(row['Opening Price'].replace('$', '').replace(',', ''))

        # For stock performance
        current_price = closing_price
        starting_price = opening_price
    
    # Determine color based on the price comparison
    if current_price > starting_price:
        color += 'green;'  # Green for positive change
    elif current_price < starting_price:
        color += 'red;'  # Red for negative change
    else:
        color += 'black;'  # No change

    return [color] * len(row)

@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data using yfinance for the given ticker and date range."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

@st.cache_data
def fetch_tickers_by_index(selected_index):
    """Fetch stock tickers based on the selected stock market index."""
    if selected_index == 'NASDAQ-100':
        return fetch_nasdaq100_tickers()
    elif selected_index == 'S&P 500':
        return fetch_sp500_tickers()
    elif selected_index == 'Dow Jones I.A.':
        return fetch_dow_jones_tickers()  # Update this to the specific function if needed
    elif selected_index == 'Dow Jones Transportation Average':
        return fetch_dow_jones_transportation_tickers()
    elif selected_index == 'Dow Jones Utility Average':
        return fetch_dow_jones_utility_tickers()
    # Add additional conditions for other indices here as needed
    else:
        return []


def calculate_statistics(data):
    """Calculate mean price and volatility."""
    mean_price = data['Close'].mean()
    volatility = data['Close'].pct_change().std() * 100  # Annualized volatility in percentage
    return mean_price, volatility

def show_page():
    st.markdown('<h1 style="text-align: center;">Stock Market Index Overview</h1>', unsafe_allow_html=True)
    st.write("")

    st.sidebar.header("User Inputs")

    # Create a selectbox for stock options
    selected_index = st.sidebar.selectbox("Select Index:", list(stock_options.keys()))

    # Checkbox to show stock information
    show_info = st.sidebar.checkbox('Show Index Information', key='show_info_checkbox')

    st.sidebar.subheader("Select Time Period")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"), key='start_date')
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"), key='end_date')

    selected_ticker, exchange_description = stock_options[selected_index]

    df_index_info = pd.DataFrame({
        'Field': ['Symbol', 'Name', 'Exchange Description'],
        'Value': [selected_ticker, selected_index, exchange_description]
    })

    cel01, cell02 = st.columns((2, 2), gap='small')
    with cel01:
        st.write(f"### {selected_index} Data")
    with cell02:
        st.write(f"### Top and Worst {selected_index} Performers")

    if start_date < end_date:
        index_data = fetch_stock_data(selected_ticker, start_date, end_date)


        if show_info:
            st.write("### Index Information")
            st.table(df_index_info.set_index('Field'))



        index_performance_data = []
        for date in index_data.index:
            closing_price = index_data['Close'][date]
            opening_price = index_data['Open'][date]
            high_price = index_data['High'][date]
            low_price = index_data['Low'][date]
            volume = index_data['Volume'][date]

            if index_data['Close'][index_data.index[0]] != 0:
                percentage_change = (closing_price - index_data['Close'][index_data.index[0]]) / index_data['Close'][index_data.index[0]] * 100
            else:
                percentage_change = 0

            index_performance_data.append({
                "Date": date.date(),
                "Opening Price": f"${opening_price:.2f}",
                "Closing Price": f"${closing_price:.2f}",
                "High": f"${high_price:.2f}",
                "Low": f"${low_price:.2f}",
                "Volume": volume,
                "% Change": f"{percentage_change:.2f}%"
            })

        df_index_performance = pd.DataFrame(index_performance_data)
        styled_df = df_index_performance.style.apply(apply_performance_color, axis=1)

        mean_price, volatility = calculate_statistics(index_data)

        period_high = index_data['Close'].max()
        period_low = index_data['Close'].min()


        # Lists all the stocks in the selected equity index
        stock_tickers = fetch_tickers_by_index(selected_index)
        stock_data_list = []
        for ticker in stock_tickers:
            stock_data = fetch_stock_data(ticker, start_date, end_date)
            if not stock_data.empty:
                current_price = stock_data['Close'].iloc[-1]  # Latest closing price
                starting_price = stock_data['Close'].iloc[0]  # Starting price at the beginning of the period
                average_price = stock_data['Close'].mean()  # Mean price calculation
                total_volume = stock_data['Volume'].sum()  # Total volume over the period

                # Calculate daily returns and volatility
                stock_data['Returns'] = stock_data['Close'].pct_change()
                volatility = stock_data['Returns'].std() * 100  # Annualized volatility in percentage

                stock_data_list.append({
                    "Ticker": ticker,
                    "Current Price": f"${current_price:.2f}",  # Format current price
                    "Starting Price": f"${starting_price:.2f}",  # Format starting price
                    "Average Price": f"${average_price:.2f}",  # Format average price
                    "% Change": f"{((current_price - starting_price) / starting_price) * 100:.2f}%",  # Percentage change
                    "Volume": f"{total_volume:,}",  # Format volume with commas
                    "Volatility (%)": f"{volatility:.2f}%"  # Format volatility
                })

        df_stocks = pd.DataFrame(stock_data_list)


        cel11, cell12 = st.columns((2, 2), gap='small')
        with cel11:

            if not index_data.empty:
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    hide_stock_data = st.checkbox('Hide Stock Data', value=False, key='hide_stock_data')

                with col2:
                    sma = st.checkbox('Show SMA', key='show_sma_checkbox')
                    if sma:
                        sma_period = st.slider('SMA Period (Business Days)', min_value=5, max_value=500, value=20, step=1, key='sma_period_slider')

                with col3:
                    sma2 = st.checkbox('Show SMA2', key='show_sma2_checkbox')
                    if sma2:
                        sma2_period = st.slider('SMA2 Period (Business Days)', min_value=5, max_value=500, value=20, step=1, key='sma2_period_slider')


                fig = go.Figure()
                fig.add_trace(go.Scatter(x=index_data.index, y=index_data['Close'], mode='lines', name=f'{selected_index} Close Price'))

                if sma:
                    index_data[f'SMA {sma_period}'] = index_data['Close'].rolling(sma_period).mean()
                    fig.add_trace(go.Scatter(x=index_data.index, y=index_data[f'SMA {sma_period}'], mode='lines', name=f'SMA {sma_period}'))

                if sma2:
                    index_data[f'SMA2 {sma2_period}'] = index_data['Close'].rolling(sma2_period).mean()
                    fig.add_trace(go.Scatter(x=index_data.index, y=index_data[f'SMA2 {sma2_period}'], mode='lines', name=f'SMA2 {sma2_period}'))

                fig.update_layout(title=f'{selected_index} Stock Price Development',
                                xaxis_title='Date',
                                yaxis_title='Price (USD)',
                                hovermode='x unified')

                st.plotly_chart(fig, use_container_width=True)            

            with cell12:

                if not df_stocks.empty:
                    show_constituents_data = st.checkbox('Hide Constituents Data', value=False, key='show_constituents_data')


                    # Calculate percentage change for the selected period
                    df_stocks['% Change'] = df_stocks['% Change'].str.replace('%', '').astype(float)

                    # Identify top 3 highest performers and bottom 3 lowest performers for selected period
                    bottom_performers = df_stocks.nsmallest(3, '% Change')
                    top_performers = df_stocks.nlargest(3, '% Change').sort_values('% Change', ascending=True)

                    # Combine performers, making sure bottom performers are on top
                    combined_performers = pd.concat([bottom_performers, top_performers])  # Losses first, gains second

                    # Create the main bar chart for the selected period
                    fig_bar_selected = go.Figure()
                    fig_bar_selected.add_trace(go.Bar(
                        y=combined_performers['Ticker'],
                        x=combined_performers['% Change'],
                        orientation='h',
                        marker_color=['red'] * len(bottom_performers) + ['green'] * len(top_performers),  # Adjust color list based on counts
                        name='Performance',
                        opacity=0.8,  # Set opacity for the bars
                        text=[f"{val:.2f}%" for val in combined_performers['% Change']],  # Show actual % values with the % sign
                        textposition='auto',  # Automatically position text
                        textfont=dict(color='white', size=13)  # Text color set to white
                    ))

                    fig_bar_selected.update_layout(
                        title=dict(text='Selected Period', xanchor='right', x=1),
                        xaxis_title='',  # Remove x-axis title
                        yaxis_title='',  # Remove y-axis title
                        showlegend=False
                    )

                    # Display the same bar chart three times in a row
                    col1, col2, col3 = st.columns(3)  # Create three columns

                    with col1:
                        # Calculate the performance based on the last 7 days
                        last_7_days_data = []
                        for ticker in stock_tickers:
                            stock_data = fetch_stock_data(ticker, end_date - pd.Timedelta(days=7), end_date)
                            if not stock_data.empty:
                                current_price = stock_data['Close'].iloc[-1]  # Latest closing price
                                starting_price = stock_data['Close'].iloc[0]  # Starting price 7 days ago
                                
                                last_7_days_data.append({
                                    "Ticker": ticker,
                                    "Current Price": current_price,
                                    "Starting Price": starting_price,
                                    "% Change": f"{((current_price - starting_price) / starting_price) * 100:.2f}%",  # Percentage change over 7 days
                                })

                        df_last_7_days = pd.DataFrame(last_7_days_data)

                        if not df_last_7_days.empty:
                            df_last_7_days['% Change'] = df_last_7_days['% Change'].str.replace('%', '').astype(float)
                            bottom_performers_7_days = df_last_7_days.nsmallest(3, '% Change')
                            top_performers_7_days = df_last_7_days.nlargest(3, '% Change').sort_values('% Change', ascending=True)
                            combined_performers_7_days = pd.concat([bottom_performers_7_days, top_performers_7_days])  # Losses first, gains second

                            # Create a new bar chart for the last 7 days
                            fig_bar_7_days = go.Figure()
                            fig_bar_7_days.add_trace(go.Bar(
                                y=combined_performers_7_days['Ticker'],
                                x=combined_performers_7_days['% Change'],
                                orientation='h',
                                marker_color=['red'] * len(bottom_performers_7_days) + ['green'] * len(top_performers_7_days),  # Adjust color list based on counts
                                name='Performance Last 7 Days',
                                opacity=0.8,  # Set opacity to make bars slightly transparent
                                text=[f"{val:.2f}%" for val in combined_performers_7_days['% Change']],  # Show actual % values with the % sign
                                textposition='auto',  # Automatically position text
                                textfont=dict(color='white', size=13)  # Text color set to white
        
                            ))

                            fig_bar_7_days.update_layout(
                                title=dict(text='Last 7 Days', xanchor='right', x=1),
                                xaxis_title='',  # Remove x-axis title
                                yaxis_title='',  # Remove y-axis title
                                showlegend=False
                            )

                            st.plotly_chart(fig_bar_7_days, use_container_width=True)

                    with col2:
                        # Calculate the performance based on the last 30 days
                        last_30_days_data = []
                        for ticker in stock_tickers:
                            stock_data = fetch_stock_data(ticker, end_date - pd.Timedelta(days=30), end_date)
                            if not stock_data.empty:
                                current_price = stock_data['Close'].iloc[-1]  # Latest closing price
                                starting_price = stock_data['Close'].iloc[0]  # Starting price 30 days ago
                                
                                last_30_days_data.append({
                                    "Ticker": ticker,
                                    "Current Price": current_price,
                                    "Starting Price": starting_price,
                                    "% Change": f"{((current_price - starting_price) / starting_price) * 100:.2f}%",  # Percentage change over 30 days
                                })

                        df_last_30_days = pd.DataFrame(last_30_days_data)

                        if not df_last_30_days.empty:
                            df_last_30_days['% Change'] = df_last_30_days['% Change'].str.replace('%', '').astype(float)
                            bottom_performers_30_days = df_last_30_days.nsmallest(3, '% Change')
                            top_performers_30_days = df_last_30_days.nlargest(3, '% Change').sort_values('% Change', ascending=True)
                            combined_performers_30_days = pd.concat([bottom_performers_30_days, top_performers_30_days])  # Losses first, gains second

                            # Create a new bar chart for the last 30 days
                            fig_bar_30_days = go.Figure()
                            fig_bar_30_days.add_trace(go.Bar(
                                y=combined_performers_30_days['Ticker'],
                                x=combined_performers_30_days['% Change'],
                                orientation='h',
                                marker_color=['red'] * len(bottom_performers_30_days) + ['green'] * len(top_performers_30_days),  # Adjust color list based on counts
                                name='Performance Last 30 Days',
                                opacity=0.8,  # Set opacity to make bars slightly transparent
                                text=[f"{val:.2f}%" for val in combined_performers_30_days['% Change']],  # Show actual % values with the % sign
                                textposition='auto',    # Automatically position text
                                textfont=dict(color='white', size=13)  # Text color set to white
                            ))

                            fig_bar_30_days.update_layout(
                                title=dict(text='Last 30 Days', xanchor='right', x=1),
                                xaxis_title='',  # Remove x-axis title
                                yaxis_title='',  # Remove y-axis title
                                showlegend=False
                            )

                            st.plotly_chart(fig_bar_30_days, use_container_width=True)

                    with col3:
                        # Display the bar chart for the selected period again
                        st.plotly_chart(fig_bar_selected, use_container_width=True)
                        styled_df_stocks = df_stocks.style.apply(lambda row: apply_performance_color(row, is_stock=True), axis=1)

                else:
                    st.error("No data found for the selected index. Please try another date range.")

        cell21, cell22 = st.columns((2, 2), gap='small')
        with cell21:
            col5, col6 = st.columns(2)
            with col5:

                final_price = index_data['Close'].iloc[-1]
                initial_price = index_data['Close'].iloc[0]
                performance_change = (final_price - initial_price) / initial_price * 100

                st.metric(
                    label="Closing Price",
                    value=f"${final_price:.2f}",
                    delta=f"{performance_change:.2f}%",
                    delta_color="normal")

            with col6:
                st.metric("Period High", f"${period_high:.2f}")

        with cell22:
            col7, col8 = st.columns(2)
            with col7:
                st.metric("Period Low", f"${period_low:.2f}")
            with col8:
                st.metric("Price Volatility (Std. Dev.)", f"{volatility:.2f}%")


        cell31, cell32 = st.columns((2, 2), gap='small')
        with cell31:
            if not hide_stock_data:
                st.write("### Index Performance")
                st.dataframe(styled_df, height=350)

        with cell32:
            if not  show_constituents_data:
                st.write("### Stocks Performance")
                styled_df_stocks = df_stocks.style.apply(lambda row: apply_performance_color(row, is_stock=True), axis=1)
                st.dataframe(styled_df_stocks, height=350)

    else:
        st.error("Error: End date must be after start date.")


    # About section
    with st.expander(':information_source: :orange[About]', expanded=False):
        st.write("- :orange[**Purpose**]: This dashboard provides a comprehensive overview of major stock market indices, enabling users to track the performance of selected indices and their constituent stocks over specified periods. It serves as a tool for financial analysis and decision-making, offering insights into market trends and volatility.")
        cell41, cell42 = st.columns((2, 2), gap='small')
        with cell41:
            st.write('''
                - :orange[**Used Metrics**]:
                    - **Closing Price**: The last price at which the stock or index traded during the regular market session.
                    - **Percentage Change**: The difference in the stock or index price from the start to the end of the selected period, expressed as a percentage.
                    - **Volatility (Std. Dev.)**: A measure of the price fluctuation, calculated as the standard deviation of daily returns, annualized to reflect the volatility over a year.
                    - **Period High/Low**: The highest and lowest prices recorded during the selected period.
                    - **Volume**: The total number of shares traded over the selected period.
                - :orange[**Additional Features**]: Users can toggle the visibility of various data elements, including specific moving averages and detailed stock performance metrics, to customize their view of the market analysis.
            ''') 
        with cell42:
            st.write('''
                - :orange[**Visualizations**]:
                    - **Stock Price Development Plot**: A line chart illustrating the closing prices of the selected index over the specified period, with optional overlays for Simple Moving Averages (SMA).
                    - **Performance Bar Charts**: Horizontal bar charts showing the top and bottom-performing stocks within the selected index for the overall period, as well as for the last 7 and 30 days.
                    - **Stock and Index Performance Tables**: Tabular data representations highlighting key performance metrics, with conditional formatting to visualize gains and losses at a glance.
                - :orange[**Data Source**]: All stock and index data is fetched from Yahoo Finance via the yfinance library.
            ''') 

