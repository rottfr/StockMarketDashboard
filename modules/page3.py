import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
import statsmodels.api as sm

@st.cache_data
def fetch_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    sp500_table = table[0]
    sp500_tickers = sp500_table['Symbol'].tolist()
    return sp500_tickers

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data using yfinance for the given ticker and date range."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def fetch_ticker_info(ticker):
    """Fetch additional information about the ticker."""
    ticker_meta = yf.Ticker(ticker)
    return {
        'Symbol': ticker_meta.info.get('symbol', 'N/A'),
        'Name': ticker_meta.info.get('shortName', 'N/A'),
        'Exchange': ticker_meta.info.get('exchange', 'N/A'),
        'Currency': ticker_meta.info.get('financialCurrency', 'N/A'),
        'Sector': ticker_meta.info.get('sector', 'N/A'),
        'Employees': ticker_meta.info.get('fullTimeEmployees', 'N/A'),
        'Market Cap': ticker_meta.info.get('marketCap', 'N/A'),
        'PEG Ratio': ticker_meta.info.get('pegRatio', 'N/A'),  # Added PEG Ratio
        'Return on Equity (ROE)': ticker_meta.info.get('returnOnEquity', 'N/A'),  # Added ROE
        'Website': ticker_meta.info.get('website', 'N/A'),
    }

@st.cache_data
def collect_basic_info(ticker):
    yf_ticker = yf.Ticker(ticker)
    basic_info = yf_ticker.get_info()['longBusinessSummary']
    return basic_info

def format_market_cap(market_cap, currency):
    """Format market cap with proper punctuation and currency."""
    if market_cap is not None:
        formatted_cap = f"{market_cap:,.0f} {currency}"  # Format with commas and currency
        return formatted_cap
    return "N/A"

def calculate_statistics(data):
    """Calculate key statistics from stock data."""
    mean_price = data['Close'].mean()
    volatility = data['Close'].std()
    return mean_price, volatility

def calculate_daily_returns(data):
    """Calculate daily returns from stock data."""
    returns = data['Close'].pct_change().dropna()
    return returns

def perform_regression(stock_returns, market_returns):
    """Perform linear regression of stock returns against market returns."""
    X = sm.add_constant(market_returns)  # Add a constant (intercept) to the model
    model = sm.OLS(stock_returns, X).fit()
    beta = model.params[1]
    adj_r_squared = model.rsquared_adj
    return beta, adj_r_squared, model

def apply_color(row):
    """Apply text color to the row based on closing price relative to opening price."""
    color = 'color: '
    if row['Close'] > row['Open']:
        color += 'green;'  # Green for positive change
    elif row['Close'] < row['Open']:
        color += 'red;'  # Red for negative change
    else:
        color += 'black;'  # No change
    return [color] * len(row)


def show_page():
    st.markdown('<h1 style="text-align: center;">S&P 500 Stock Market Overview</h1>', unsafe_allow_html=True)
    st.write("")

    st.sidebar.header("User Inputs")

    available_tickers = fetch_sp500_tickers()

    ticker = st.sidebar.selectbox("Select Stock Ticker Symbol:", options=available_tickers)

    # Fetch stock data to display the current closing price and performance change
    data = fetch_stock_data(ticker, pd.to_datetime("2023-01-01"), pd.to_datetime("today"))

    if not data.empty:
        final_price = data['Close'].iloc[-1]
        initial_price = data['Close'].iloc[0]
        performance_change = (final_price - initial_price) / initial_price * 100

        # Adjust the delta_color based on performance_change
        if performance_change > 0:
            delta_color = "normal"  # Positive change - Green
        elif performance_change < 0:
            delta_color = "inverse"  # Negative change - Red
        else:
            delta_color = "off"      # No change - Default color

    # Checkboxes for displaying company info and hiding stock data on the same line
    show_info = st.sidebar.checkbox('Show Stock Info')

    st.sidebar.subheader("Select Time Period")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    st.write("")

    if start_date < end_date:
        # Fetch stock data and S&P 500 data
        data = fetch_stock_data(ticker, start_date, end_date)
        sp500_data = fetch_stock_data("^GSPC", start_date, end_date)

        # Format the index to show only the date
        data.index = data.index.date  # Convert datetime index to date only
        sp500_data.index = sp500_data.index.date  # Convert datetime index to date only


        if show_info:
   
            ticker_info = fetch_ticker_info(ticker)
            if ticker_info['Market Cap'] != 'N/A':
                ticker_info['Market Cap'] = format_market_cap(ticker_info['Market Cap'], ticker_info['Currency'])
            info_df = pd.DataFrame.from_dict(ticker_info, orient='index', columns=['Value']).reset_index()
            info_df.columns = ['Basic Info', 'Value']
            info_df = info_df[info_df['Basic Info'].isin(['Symbol', 'Name', 'Exchange', 'Employees', 'Sector', 'Market Cap', 'PEG Ratio', 'Return on Equity (ROE)', 'Website'])]
            info_df = info_df.set_index('Basic Info').transpose()

            st.write("### Stock Information")
            # Display the table with bold headers and without the index

            ticker_info_two = collect_basic_info(ticker)
            st.write(f"{collect_basic_info(ticker)}")

            st.markdown(
                info_df.style.set_table_styles(
                    [{'selector': 'th', 'props': [('font-weight', 'bold')]}]  # Make header text bold
                ).hide(axis='index').to_html(index=False, header=True, border=0),
                unsafe_allow_html=True
            )

            
        if not data.empty and not sp500_data.empty:
            
            st.write("")

            stock_returns = calculate_daily_returns(data)
            market_returns = calculate_daily_returns(sp500_data)

            standardized_stock_returns = (stock_returns - stock_returns.mean()) / stock_returns.std()
            standardized_market_returns = (market_returns - market_returns.mean()) / market_returns.std()

            # Perform regression analysis
            beta, adj_r_squared, model = perform_regression(stock_returns, market_returns)
            alpha = model.params[0]

            cell01, cell02 = st.columns((2, 2), gap='small')
            with cell01:
                st.markdown(f'<h3 style="text-align: center;">{ticker} Stock Data</h3>', unsafe_allow_html=True)
            with cell02:
                st.markdown(f'<h3 style="text-align: center;">Relationship Between {ticker} and S&P 500 Returns</h3>', unsafe_allow_html=True)
            

            cell11, cell12 = st.columns((2, 2), gap='small')
            with cell11:

                # Plot the closing price of the stock
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

                # Create columns for SMA and SMA2 checkboxes
                col2, col3, col4 = st.columns(3)
                with col2:
                    show_stock_data = st.checkbox('Hide Stock Data', value=False)
                with col3:
                    sma = st.checkbox('Show SMA')
                    if sma:
                        period = st.slider('SMA Period (Business Days)', min_value=5, max_value=500, value=20, step=1)
                        data[f'SMA {period}'] = data['Close'].rolling(period).mean()
                        fig.add_trace(go.Scatter(x=data.index, y=data[f'SMA {period}'], mode='lines', name=f'SMA {period}'))

                with col4:
                    sma2 = st.checkbox('Show SMA2')
                    if sma2:
                        period2 = st.slider('SMA2 Period (Business Days)', min_value=5, max_value=500, value=20, step=1)
                        data[f'SMA2 {period2}'] = data['Close'].rolling(period2).mean()
                        fig.add_trace(go.Scatter(x=data.index, y=data[f'SMA2 {period2}'], mode='lines', name=f'SMA2 {period2}'))

                fig.update_layout(title=f'{ticker} Stock Price',
                                xaxis_title='Date',
                                yaxis_title='Price (USD)',
                                height=450,
                                hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

                data['% Change'] = data['Close'].pct_change().mul(100)
                data['% Change'] = data['% Change'].apply(lambda x: f"{x:.2f}%")

                mean_price, volatility = calculate_statistics(data)

                period_high = data['Close'].max()
                period_low = data['Close'].min()

                col5, col6, col7, col8 = st.columns(4)

                with col5:
                    st.metric(
                        label="Closing Price",
                        value=f"${final_price:.2f}",
                        delta=f"{performance_change:.2f}%",
                        delta_color="normal"
                    )
                    
                with col6:
                    st.metric("Mean Closing Price", f"${mean_price:.2f}")

                with col7:
                    st.metric("Period High", f"${period_high:.2f}")

                with col8:
                    st.metric("Period Low", f"${period_low:.2f}")

                st.write("")

                if not show_stock_data:
                    # Apply conditional formatting and make the dataframe scrollable
                    styled_data = data.style.apply(apply_color, axis=1)
                    st.dataframe(styled_data, height=390)


            with cell12:
                # Checkbox to hide the regression plot
                hide_regression_plot = st.checkbox('Hide Regression Plot', value=False)

                standardized_fig = go.Figure()

                standardized_fig.add_trace(go.Scatter(
                    x=standardized_stock_returns.index, 
                    y=standardized_stock_returns, 
                    mode='lines', 
                    name=f'Standardized {ticker} Returns'
                ))

                standardized_fig.add_trace(go.Scatter(
                    x=standardized_market_returns.index, 
                    y=standardized_market_returns, 
                    mode='lines', 
                    name='Standardized S&P 500 Returns'
                ))

                standardized_fig.update_layout(
                    title='Standardized Daily Returns',
                    xaxis_title='Date',
                    yaxis_title='Standardized Returns',
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=1.1, xanchor="center", x=0.5),  # Legend position at the top
                    height=450  # Increased height for larger vertical size
                )
                standardized_fig.update_traces(marker=dict(size=5))  # Increase marker size
                st.plotly_chart(standardized_fig, use_container_width=True)


                col9, col10, col11, col12 = st.columns(4)
                with col9:
                    st.metric("Price Volatility (Std. Dev.)", f"${volatility:.2f}")
                with col10:
                    st.metric(label="Beta (Exposure to S&P 500)", value=f"{beta:.4f}")
                with col11:
                    st.metric(label="Adjusted R-squared", value=f"{adj_r_squared:.4f}")
                with col12:
                    st.metric(label="Alpha (Excess Return)", value=f"{alpha * 100:.4f}%")


                if not hide_regression_plot:
                    regression_fig = go.Figure()

                    regression_fig.add_trace(go.Scatter(
                        x=market_returns, y=stock_returns, mode='markers', name='Data Points'
                    ))

                    regression_fig.add_trace(go.Scatter(
                        x=market_returns, y=model.predict(sm.add_constant(market_returns)),
                        mode='lines', name='Regression Line'
                    ))

                    regression_fig.update_layout(
                        title=f'Regression of {ticker} Returns vs. S&P 500 Returns',
                        xaxis_title='S&P 500 Daily Returns',
                        yaxis_title=f'{ticker} Daily Returns',
                        hovermode='closest'
                    )

                    st.plotly_chart(regression_fig, use_container_width=True)
        else:
            st.error("No data found for the selected stock ticker or S&P 500. Please try another ticker.")
    else:
        st.error("Error: End date must be after start date.")


    # About section
    with st.expander(':information_source: :orange[About]', expanded=False):
        st.write("- :orange[**Purpose**]: This dashboard offers a detailed analysis of S&P 500 stocks, allowing users to explore stock performance, financial metrics, and relationships with the broader market. It is designed as a tool for financial analysis and investment decision-making, providing insights into stock behavior and market trends.")
        
        cell41, cell42 = st.columns((2, 2), gap='small')
        
        with cell41:
            st.write('''
                - :orange[**Used Metrics**]:
                    - **Closing Price**: The final price at which the stock traded on a given day.
                    - **Percentage Change**: The change in the stock price over the selected period, shown as a percentage.
                    - **Volatility (Std. Dev.)**: Reflects the stock's price fluctuations, calculated as the standard deviation of daily returns over the period.
                    - **Beta**: A measure of the stock's exposure to the S&P 500, indicating its sensitivity to market movements.
                    - **Alpha**: The excess return of the stock compared to the market, adjusted for risk.
                    - **Market Cap**: The total market value of a company’s outstanding shares.
            ''')
        
        with cell42:
            st.write('''
                - :orange[**Visualizations**]:
                    - **Stock Price Chart**: A dynamic line chart displaying the stock’s closing prices over time, with options to add Simple Moving Averages (SMA).
                    - **Regression Analysis**: A scatter plot showing the relationship between the stock’s returns and the S&P 500 returns, along with a regression line to visualize correlations.
                    - **Performance Metrics**: A set of key financial indicators presented with color-coded metrics for quick interpretation of stock performance.
                    - **Detailed Data Table**: A comprehensive table with stock data, including daily price changes, formatted for easy review.
            ''')
        st.write("- :orange[**Data Source**]: All stock and index data is fetched from Yahoo Finance via the yfinance library.")
