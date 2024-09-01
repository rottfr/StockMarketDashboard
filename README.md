# Stock Market Dashboard
========================

## Overview

The Stock Market Dashboard is a web application built using Streamlit, Pandas, and yfinance that allows users to track various stock indices, retrieve ticker information, and explore industry-specific keywords. This tool provides insights into market performance and helps users navigate financial data effectively.

### Features

- **Stock Indices**: Fetches and displays tickers for major stock indices such as:
  - Dow Jones Industrial Average (DJI)
  - NASDAQ-100 (NDX)
  - S&P 500 (GSPC)
  - Dow Jones Transportation Average (DJT)
  - Dow Jones Utility Average (DJU)

- **Ticker Retrieval**: Automatically retrieves ticker symbols from Wikipedia for the selected indices.

- **State Abbreviations**: Contains a comprehensive list of US states and their abbreviations, useful for filtering and organizing data.

- **Industry Keywords**: Categorizes industry-specific keywords into sectors such as Technology, Finance, Healthcare, and more, facilitating targeted financial analysis.

- **Interactive Data Visualization**: Displays stock price trends and performance metrics in interactive charts for better insights into market behavior.

- **User-friendly Interface**: Designed with a simple and intuitive user interface using Streamlit, making it easy for users to navigate and access stock information.

- **Caching for Performance**: Utilizes caching mechanisms to improve performance and reduce loading times for frequently accessed data.

- **Customizable Filters**: Allows users to filter stocks based on various criteria, including industry sector and state abbreviation, enabling more precise analyses.

- **Real-time Data Updates**: Fetches real-time stock prices and data from Yahoo Finance, ensuring users have the latest information at their fingertips.

- **Data Export Options**: Provides options to export stock data in various formats, making it easy for users to analyze data offline or share it with others.

- **Comprehensive Documentation**: Includes detailed documentation and tutorials to help users understand the functionalities and maximize their usage of the application.

- **Responsive Design**: Optimized for use on various devices, ensuring a consistent experience across desktops, tablets, and mobile devices.


## Installation

To run this application, you'll need Python (Created with 3.12.5)


**How to run this demo**

1.  cd to the directory where *requirements.txt* is located

2.  activate your virualenv

3.  run: `pip install -r requirements.txt` in your shell

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
git clone https://github.com/rottfr/StockMarketDashboard.git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 
=

cd into the project folder

 

'path'\>cd StockMarketDashboard

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

streamlit run app.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
