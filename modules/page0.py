import streamlit as st
from PIL import Image

def show_page():

    st.markdown('<h1 style="text-align: center;">Welcome!</h1>', unsafe_allow_html=True)
    st.markdown('<h5 style="text-align: center;">This App contains an Overview of financial market using multiple analysis tools. Please select the corresponding dashboard in the Navigation bar.</h5>', unsafe_allow_html=True)


    # Create two columns with medium gap
    col1, col2 = st.columns((2, 2), gap='medium')

    # Column 1
    with col1:
        st.markdown('<h3 style="text-align: center;">Market Overview</h3>', unsafe_allow_html=True)
        image1 = Image.open("Images/stocks.jpg")
        st.image(image1)
        st.markdown('<h5 style="text-align: center;">A general stock market index where you can investigate equities and their listed stocks.</h5>', unsafe_allow_html=True)
        
    # Column 2
    with col2:
        st.markdown('<h3 style="text-align: center;">S&P 500 Stock Market Overview</h3>', unsafe_allow_html=True)
        image3 = Image.open("Images/stocks2.jpeg")
        st.image(image3)
        st.markdown('<h5 style="text-align: center;">Panel of specific metrics for S&P500 listed stocks and their relation to the market index.</h5>', unsafe_allow_html=True)

    col3, col4 = st.columns((2, 2), gap='medium')

    with col3:
        st.markdown('<h3 style="text-align: center;">Financial News Sentiment Analysis</h3>', unsafe_allow_html=True)
        image2 = Image.open("Images/stocks3.jpg")
        st.image(image2)
        st.markdown('<h5 style="text-align: center;">A Dashboard that brings you the latest financial news/sentiment analysis.</h5>', unsafe_allow_html=True)

    with col4:
        st.markdown('<h3 style="text-align: center;">S&P 500 Sector Analysis</h3>', unsafe_allow_html=True)
        image4 = Image.open("Images/stocks1.jpg")
        st.image(image4)
        st.markdown('<h5 style="text-align: center;">An overview of the development in different industries of S&P 500 listed stocks.</h5>', unsafe_allow_html=True)
        

