import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import yfinance as yf
import plotly.graph_objects as go
from modules.utils import states_abbreviation, stock_options, fetch_sp500_tickers, fetch_nasdaq100_tickers


@st.cache_data
def fetch_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    return tables[0]

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data using yfinance for the given ticker and date range."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def make_performance_heatmap(data, x_col, y_col, color_col, color_scheme):
    """Create a heatmap for stock price performance."""
    heatmap = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            color=alt.Color(color_col, scale=alt.Scale(scheme=color_scheme)),
            tooltip=[x_col, y_col, color_col]
        )
        .properties(
            width=400,
            height=450,
        )
    )
    return heatmap

@st.cache_data
def load_and_prepare_data():
    sp500_data = fetch_sp500()
    sp500_data['Date added'] = pd.to_datetime(sp500_data['Date added'])
    sp500_data.columns = sp500_data.columns.str.strip()
    sp500_data['State'] = sp500_data['Headquarters Location'].str.split(', ').str[-1]

    # Precompute the sector performance based on the full dataset
    stock_data_list_copy = []
    stock_tickers_copy = sp500_data['Symbol'].tolist()

    for ticker in stock_tickers_copy:
        stock_data = fetch_stock_data(ticker, pd.to_datetime("2023-01-01"), pd.to_datetime("today"))
        if not stock_data.empty:
            current_price = stock_data['Close'].iloc[-1]
            starting_price = stock_data['Close'].iloc[0]
            stock_data_list_copy.append({
                "Ticker": ticker,
                "Current Price": current_price,
                "Starting Price": starting_price,
                "% Change": ((current_price - starting_price) / starting_price) * 100,
                "Sector": sp500_data[sp500_data['Symbol'] == ticker]['GICS Sector'].values[0]
            })

    df_stock_performance_copy = pd.DataFrame(stock_data_list_copy)
    df_stock_performance_copy['% Change'] = df_stock_performance_copy['% Change'].round(2)

    sector_performance = df_stock_performance_copy.groupby("Sector")["% Change"].mean().reset_index()
    sector_performance = sector_performance.sort_values(by="% Change", ascending=True)

    return sp500_data, sector_performance

def show_page():
    st.markdown('<h1 style="text-align: center;">S&P 500 Sector Growth Overview</h1>', unsafe_allow_html=True)
    st.write("")

    # Load data and sector performance (cached)
    sp500_data, sector_performance = load_and_prepare_data()

    # Define sectors for filtering
    sectors = sp500_data['GICS Sector'].unique().tolist()
    sectors.insert(0, "All")

    # Sidebar
    with st.sidebar:
        selected_sector = st.selectbox('Select a sector', sectors)

        st.sidebar.subheader("Select Time Period")
        start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"), key='start_date')
        end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"), key='end_date')

        if selected_sector != "All":
            filtered_data = sp500_data[sp500_data['GICS Sector'] == selected_sector]
        else:
            filtered_data = sp500_data

        # Recalculate stock performance for the selected sector based on time period
        stock_data_list = []
        stock_tickers = filtered_data['Symbol'].tolist()

        for ticker in stock_tickers:
            stock_data = fetch_stock_data(ticker, start_date, end_date)
            if not stock_data.empty:
                current_price = stock_data['Close'].iloc[-1]
                starting_price = stock_data['Close'].iloc[0]
                stock_data_list.append({
                    "Ticker": ticker,
                    "Current Price": current_price,
                    "Starting Price": starting_price,
                    "% Change": ((current_price - starting_price) / starting_price) * 100,
                    "Sector": filtered_data[filtered_data['Symbol'] == ticker]['GICS Sector'].values[0]
                })

        filtered_state_counts = filtered_data['State'].value_counts().reset_index()
        filtered_state_counts.columns = ['State', 'Number of Companies']
        filtered_state_counts['State Abbreviation'] = filtered_state_counts['State'].map(states_abbreviation)

        filtered_missing_abbreviations = filtered_state_counts[filtered_state_counts['State Abbreviation'].isna()]
        filtered_missing_abbreviations = filtered_missing_abbreviations.drop(columns=['State Abbreviation'])

        filtered_state_counts = filtered_state_counts[filtered_state_counts['State'].isin(states_abbreviation.keys())]

        color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
        selected_color_theme = st.selectbox('Select a color theme', color_theme_list, index=color_theme_list.index('inferno'))


    st.markdown('<h3 style="text-align: center;">Stock Growth per State and Sector</h3>', unsafe_allow_html=True)
    # Main dashboard layout
    col1, col2, col3 = st.columns((2, 4.15, 2), gap='small')
    # Section 1: Best and Worst Performers
    with col1:
        st.markdown('#### Best and Worst Performers')
        if stock_data_list:  # Ensure stock data exists
            df_stock_performance = pd.DataFrame(stock_data_list)
            df_stock_performance['% Change'] = df_stock_performance['% Change'].round(2)

            # Identify top 3 highest performers and bottom 3 lowest performers for selected period
            bottom_performers = df_stock_performance.nsmallest(5, '% Change')
            top_performers = df_stock_performance.nlargest(5, '% Change').sort_values('% Change', ascending=True)

            # Combine performers, making sure bottom performers are on top
            combined_performers = pd.concat([bottom_performers, top_performers])

            # Create the bar chart
            fig_bar_selected = go.Figure()
            fig_bar_selected.add_trace(go.Bar(
                y=combined_performers['Ticker'],
                x=combined_performers['% Change'],
                orientation='h',
                marker_color=['red'] * len(bottom_performers) + ['green'] * len(top_performers),
                name='Performance',
                opacity=0.8,
                text=[f"{val:.2f}%" for val in combined_performers['% Change']],
                textposition='auto',
                textfont=dict(color='white', size=13)
            ))

            fig_bar_selected.update_layout(
                xaxis_title='',
                yaxis_title='',
                showlegend=False,
                plot_bgcolor='rgba(255, 255, 255, 0)',
                paper_bgcolor='rgba(255, 255, 255, 0)',
                margin=dict(l=0, r=0, t=30, b=30)
            )

            st.plotly_chart(fig_bar_selected, use_container_width=True)
        else:
            st.write("No performance data available to display the best and worst performers.")

    # Section 2: Stock Growth per State and Sector
    with col2:
        st.markdown('<h4 style="text-align: center;">Closing Price Growth</h4>', unsafe_allow_html=True)
        if stock_data_list:  # Ensure data exists before creating DataFrame
            df_stock_performance = pd.DataFrame(stock_data_list)
            df_stock_performance['% Change'] = df_stock_performance['% Change'].round(2)

            heatmap = make_performance_heatmap(df_stock_performance, 'Ticker', '% Change', '% Change', selected_color_theme)
            st.altair_chart(heatmap, use_container_width=True)
        else:
            st.write("No stock performance data available for the selected sector.")

    # Section 3: Average % Change by Sector
    with col3:
        st.markdown('#### Avg. % Change by Sector')
        fig_bar_sectors = go.Figure()
        fig_bar_sectors.add_trace(go.Bar(
            y=sector_performance['Sector'],
            x=sector_performance['% Change'],
            orientation='h',
            marker_color=['green' if x > 0 else 'red' for x in sector_performance['% Change']],
            name='Performance',
            opacity=0.8,
            text=[f"{val:.2f}%" for val in sector_performance['% Change']],
            textposition='auto',
            textfont=dict(color='white', size=13)
        ))

        fig_bar_sectors.update_layout(
            xaxis_title='',
            yaxis_title='',
            showlegend=False,
            plot_bgcolor='rgba(255, 255, 255, 0)',
            paper_bgcolor='rgba(255, 255, 255, 0)',
            margin=dict(l=0, r=0, t=30, b=30)
        )

        st.plotly_chart(fig_bar_sectors, use_container_width=True)

    # Second row for US states, choropleth and other dataframes
    col4, col5, col6 = st.columns((2, 4.15, 2), gap='small')

    # Section 1: Top US States
    with col4:
        st.markdown('<h3 style="text-align: center;"></h3>', unsafe_allow_html=True)
        st.markdown('#### Top US States')
        st.dataframe(filtered_state_counts,
                     column_order=("State", "Number of Companies"),
                     hide_index=True,
                     column_config={
                        "State": st.column_config.TextColumn("State"),
                        "Number of Companies": st.column_config.ProgressColumn(
                            "Number of Companies",
                            format="%d",
                            min_value=0,
                            max_value=max(filtered_state_counts["Number of Companies"]),
                        )}
                     )

    # Section 2: S&P 500 Companies by State
    with col5:
        st.markdown('<h3 style="text-align: center;">State Analysis</h3>', unsafe_allow_html=True)
        st.markdown('<h4 style="text-align: center;">S&P 500 Companies by State</h4>', unsafe_allow_html=True)
        choropleth = px.choropleth(
            filtered_state_counts,
            locations='State Abbreviation',
            locationmode="USA-states",
            color='Number of Companies',
            color_continuous_scale=selected_color_theme,
            scope="usa",
            labels={'Number of Companies': 'Number of Companies'},
        )

        choropleth.update_geos(
            bgcolor="rgba(255, 255, 255, 0)",
            landcolor="rgba(255, 255, 255, 0)",
            showland=True,
            showcountries=True,
            coastlinecolor="grey",
            countrycolor="grey"
        )

        choropleth.update_layout(
            plot_bgcolor='rgba(255, 255, 255, 0)',
            paper_bgcolor='rgba(255, 255, 255, 0)'
        )

        st.plotly_chart(choropleth, use_container_width=True)

    # Section 3: Other Countries
    with col6:
        st.markdown('<h3 style="text-align: center;"></h3>', unsafe_allow_html=True)
        st.markdown('#### Other Countries')
        st.dataframe(filtered_missing_abbreviations,
                     column_order=("State", "Number of Companies"),
                     hide_index=True,
                     column_config={
                        "State": st.column_config.TextColumn("Country"),
                        "Number of Companies": st.column_config.ProgressColumn(
                            "Number of Companies",
                            format="%d",
                            min_value=0,
                            max_value=max(filtered_missing_abbreviations["Number of Companies"]) if not filtered_missing_abbreviations.empty else 0,
                        )}
                     )
                     
        st.metric(label="Number of Companies in Sector", value=filtered_data.shape[0])
        
    # About section
    with st.expander(':information_source: :orange[About]', expanded=False):
        st.write("""
        - :orange[**Purpose**]: This dashboard provides a comprehensive overview of S&P 500 sector performance, 
        enabling users to analyze stock performance, sector trends, and geographic distributions. It is a valuable tool for 
        investors and analysts seeking insights into market behavior and investment opportunities. While the current approach 
        only considering the headquarters location may be naive, it also allows the investiagtion of companies that may utilize 
        tax havens and where those are located.
        """)

        cell41, cell42 = st.columns((2, 2), gap='small')

        with cell41:
            st.write('''
                - :orange[**Used Metrics**]:
                    - **Current Price**: The latest closing price of the stock.
                    - **Starting Price**: The price at the beginning of the selected period.
                    - **Percentage Change**: The relative change in stock price during the selected period, expressed as a percentage.
                    - **Sector Performance**: Average percentage change grouped by sector, highlighting the performance of different sectors within the S&P 500.
                    - **State Distribution**: The number of companies located in each state, providing geographical context to sector performance.
            ''')

        with cell42:
            st.write('''
                - :orange[**Visualizations**]:
                    - **Best and Worst Performers**: A horizontal bar chart showcasing the top and bottom performers within the selected sector and time frame.
                    - **Stock Growth Heatmap**: An interactive heatmap visualizing the percentage change for each stock, categorized by sector.
                    - **Average % Change by Sector**: A bar chart illustrating the average performance of each sector, providing a quick overview of sector trends.
                    - **Choropleth Map**: A geographical representation of the number of S&P 500 companies by state, highlighting regional distributions.
            ''')
        
        st.write("- :orange[**Data Source**]: All stock data and performance metrics are obtained from Yahoo Finance using the yfinance library and the S&P 500 company list from Wikipedia.")

