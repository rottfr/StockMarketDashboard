import pandas as pd
import yfinance as yf
import streamlit as st

stock_options = {
    "Dow Jones I.A.": ("^DJI", "A stock market index that indicates the value of 30 large, publicly-owned companies based in the United States."),
    "NASDAQ-100": ("^NDX", "An index composed of 100 of the largest non-financial companies listed on the Nasdaq Stock Market."),
    "Dow Jones Transportation Average": ("^DJT", "Comprises 20 transportation-related stocks, providing insight into the transportation sector's performance."),
    "Dow Jones Utility Average": ("^DJU", "Includes 15 utility companies in the U.S., focusing on the performance of utility stocks."),
    "S&P 500": ("^GSPC", "A stock market index that measures the stock performance of 500 large companies listed on stock exchanges in the United States."),
}


@st.cache_data
def fetch_nasdaq100_tickers():
    url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
    table = pd.read_html(url)
    nasdaq100_table = table[4] 
    nasdaq100_tickers = nasdaq100_table['Ticker'].tolist()
    return nasdaq100_tickers
    

@st.cache_data
def fetch_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    sp500_table = table[0]
    sp500_tickers = sp500_table['Symbol'].tolist()
    return sp500_tickers

@st.cache_data
def fetch_dow_jones_tickers():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    table = pd.read_html(url)
    dow_jones_table = table[1]  
    dow_jones_tickers = dow_jones_table['Symbol'].tolist()
    return dow_jones_tickers

@st.cache_data
def fetch_dow_jones_transportation_tickers():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Transportation_Average'
    table = pd.read_html(url)
    dow_transport_table = table[0]  
    dow_transport_tickers = dow_transport_table['Ticker'].tolist()
    return dow_transport_tickers

@st.cache_data
def fetch_dow_jones_utility_tickers():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Utility_Average'
    table = pd.read_html(url)
    dow_util_table = table[1]
    dow_util_tickers = dow_util_table['Ticker'].tolist()
    return dow_util_tickers


states_abbreviation = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

industry_keywords = {
  "Technology": [
    "tech",
    "software",
    "AI",
    "cloud",
    "innovation",
    "cybersecurity",
    "data science",
    "machine learning",
    "IoT",
    "blockchain",
    "big data",
    "augmented reality",
    "virtual reality",
    "SaaS",
    "digital transformation",
    "open source",
    "DevOps",
    "5G",
    "edge computing",
    "IT infrastructure"
  ],
  "Finance": [
    "finance",
    "bank",
    "investment",
    "stocks",
    "market",
    "capital",
    "wealth management",
    "cryptocurrency",
    "financial services",
    "risk management",
    "credit",
    "loans",
    "insurance",
    "trading",
    "hedge funds",
    "venture capital",
    "equity",
    "financial planning",
    "personal finance",
    "pension funds"
  ],
  "Healthcare": [
    "health",
    "medicine",
    "hospital",
    "pharma",
    "treatment",
    "biotechnology",
    "medical devices",
    "telemedicine",
    "healthcare IT",
    "public health",
    "patient care",
    "clinical trials",
    "pharmaceuticals",
    "health insurance",
    "wellness",
    "genomics",
    "mental health",
    "healthcare analytics",
    "telehealth",
    "emergency care"
  ],
  "Energy": [
    "energy",
    "oil",
    "gas",
    "renewable",
    "solar",
    "wind",
    "energy efficiency",
    "sustainability",
    "electric vehicles",
    "nuclear energy",
    "grid",
    "clean energy",
    "biofuels",
    "carbon footprint",
    "smart grids",
    "energy storage",
    "hydropower",
    "geothermal",
    "fossil fuels",
    "energy policy"
  ],
  "Consumer Goods": [
    "consumer",
    "retail",
    "goods",
    "products",
    "brand",
    "e-commerce",
    "supply chain",
    "marketing",
    "packaging",
    "sustainability",
    "fashion",
    "electronics",
    "home goods",
    "food and beverage",
    "customer experience",
    "luxury goods",
    "wholesale",
    "merchandising",
    "private label",
    "market research"
  ],
  "Manufacturing": [
    "manufacturing",
    "production",
    "supply chain",
    "automation",
    "robotics",
    "lean manufacturing",
    "quality control",
    "materials",
    "3D printing",
    "Industry 4.0",
    "just-in-time",
    "mass production",
    "fabrication",
    "component sourcing",
    "contract manufacturing"
  ],
  "Communication": [
    "telecommunications",
    "network",
    "5G",
    "mobile",
    "wireless",
    "broadband",
    "VoIP",
    "communication",
    "satellite",
    "internet service provider",
    "telephony",
    "digital communication",
    "telecom infrastructure",
    "cable",
    "fiber optics"
  ],
  "Real Estate": [
    "real estate",
    "property",
    "investment",
    "commercial",
    "residential",
    "development",
    "brokerage",
    "mortgages",
    "leasing",
    "zoning",
    "appraisal",
    "property management",
    "real estate investment trust",
    "vacancy",
    "title insurance"
  ],
  "Transportation": [
    "transportation",
    "logistics",
    "shipping",
    "freight",
    "supply chain",
    "air travel",
    "public transport",
    "infrastructure",
    "autonomous vehicles",
    "last-mile delivery",
    "traffic management",
    "urban mobility",
    "transportation network companies",
    "rail transport"
  ],
  "Education": [
    "education",
    "learning",
    "e-learning",
    "online courses",
    "curriculum",
    "vocational training",
    "accreditation",
    "EdTech",
    "tutoring",
    "higher education",
    "adult education",
    "educational resources",
    "distance learning",
    "training programs",
    "educational technology"
  ]
}
