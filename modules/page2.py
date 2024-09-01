import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
import feedparser
import altair as alt
import yfinance as yf
import numpy as np
from modules.utils import industry_keywords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Cache the model and tokenizer loading
@st.cache_resource
def load_model():
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', clean_up_tokenization_spaces=True)
    return finbert, tokenizer

# Cache the sentiment analysis pipeline setup
@st.cache_resource
def setup_pipeline(_model, _tokenizer):
    nlp = pipeline("sentiment-analysis", model=_model, tokenizer=_tokenizer)
    return nlp

def generate_wordcloud(text):
    # Generate a word cloud image
    wordcloud = WordCloud(width=400, height=200, background_color=None, mode='RGBA').generate(text)
    
    # Create a matplotlib figure to display the word cloud
    plt.figure(figsize=(8, 4), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide the axes
    plt.tight_layout(pad=0)
    
    # Save the word cloud image to a BytesIO object
    from io import BytesIO
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    plt.close()  # Close the figure to free memory
    img.seek(0)  # Rewind the BytesIO object for reading
    return img

# Load the FinBERT model and tokenizer
finbert, tokenizer = load_model()
nlp = setup_pipeline(finbert, tokenizer)

# Caching the news feed fetching
@st.cache_data
def get_news_feed(url):
    feed = feedparser.parse(url)
    feed_df = pd.DataFrame(feed.entries)
    return feed_df

def analyze_sentiment(titles):
    sentiments = nlp(titles)
    return sentiments

# Remove some ads for wsj
def filter_titles(df, min_length=10, unwanted_titles=None):
    if unwanted_titles is None:
        unwanted_titles = []
    mask = df['title'].apply(lambda title: any(unwanted_title in title for unwanted_title in unwanted_titles))
    filtered_df = df[~mask & (df['title'].str.len() >= min_length)]
    return filtered_df

# Function to create donut chart
def make_donut(input_response, input_text):
    if input_response > 50:
        chart_color = ['#27AE60', '#12783D']  # Green
    else:
        chart_color = ['#E74C3C', '#781F16']  # Red

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100 - input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })
    
    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)
    
    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)
    return plot_bg + plot + text

# News pages
wsj_url = "https://feeds.a.dj.com/rss/RSSWorldNews.xml"
yahoo_url = 'https://finance.yahoo.com/news/rssindex'
cnbc_url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
wsj_feed = get_news_feed(wsj_url)
yahoo_feed = get_news_feed(yahoo_url)
cnbc_feed = get_news_feed(cnbc_url)

# Define unwanted titles (Ads)
unwanted_titles = ["More »", "More &raquo;", "News Quiz"] 

# Filter WSJ titles to remove very short ones and unwanted titles
wsj_feed_filtered = filter_titles(wsj_feed, unwanted_titles=unwanted_titles)
wsj_sentiments = analyze_sentiment(wsj_feed_filtered.title.tolist())

# Analyze Yahoo Finance sentiment without filtering
yahoo_feed_filtered = yahoo_feed  # No image extraction needed
yahoo_sentiments = analyze_sentiment(yahoo_feed_filtered.title.tolist())

# Analyze CNBC sentiment
cnbc_feed_filtered = cnbc_feed  # No image extraction needed
cnbc_sentiments = analyze_sentiment(cnbc_feed_filtered.title.tolist())

# Calculate matching scores based on keyword occurrences
def calculate_matching_scores(titles, industry_keywords):
    scores = {}
    for industry, keywords in industry_keywords.items():
        count = sum(any(keyword.lower() in title.lower() for keyword in keywords) for title in titles)
        scores[industry] = count * 10  # Scale the count to a score out of 100
    return scores

def show_page():
    st.markdown('<h1 style="text-align: center;">Financial News Dashboard</h1>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns((0.9, 1), gap='small')

    # News source selection in the sidebar
    st.sidebar.title("Navigation")
    source = st.sidebar.radio("Select News Source", ["All", "Wall Street Journal", "Yahoo Finance", "CNBC"])
    
    if source == "Wall Street Journal":
        feed_df = wsj_feed_filtered
        sentiments = wsj_sentiments
        source_name = "Wall Street Journal"
    elif source == "Yahoo Finance":
        feed_df = yahoo_feed_filtered
        sentiments = yahoo_sentiments
        source_name = "Yahoo Finance"
    elif source == "CNBC":
        feed_df = cnbc_feed_filtered
        sentiments = cnbc_sentiments
        source_name = "CNBC"
    else:
        feed_df = pd.concat([wsj_feed_filtered, yahoo_feed_filtered, cnbc_feed_filtered])
        all_titles = (wsj_feed_filtered.title.tolist() + 
                      yahoo_feed_filtered.title.tolist() + 
                      cnbc_feed_filtered.title.tolist())
        sentiments = analyze_sentiment(all_titles)
        source_name = "All Sources"

    # Keep all titles for word cloud generation
    all_titles_for_wordcloud = feed_df['title'].tolist()


    # Calculate overall sentiment percentages
    sentiment_labels = [sentiment['label'] for sentiment in sentiments]
    sentiment_counts = pd.Series(sentiment_labels).value_counts(normalize=True) * 100
    positive_percentage = round(sentiment_counts.get('Positive', 0))
    negative_percentage = round(sentiment_counts.get('Negative', 0))

    # Donut chart color based on positive percentage
    color = 'green' if positive_percentage > 50 else 'red'

    # Sentiment filter in the sidebar
    sentiment_filter = st.sidebar.selectbox("Filter by Sentiment", ["All", "Positive", "Negative"])
    if sentiment_filter != "All":
        filtered_news = feed_df[[sentiment['label'] == sentiment_filter for sentiment in sentiments]]
    else:
        filtered_news = feed_df

    # Calculate matching scores for the filtered news
    matching_scores = calculate_matching_scores(filtered_news['title'].tolist(), industry_keywords)

    # Create DataFrame for matching scores
    matching_scores_df = pd.DataFrame(list(matching_scores.items()), columns=['Industry', 'Matching Score'])

    st.sidebar.markdown('<h4 style="text-align: center;">Positivity Score</h4>', unsafe_allow_html=True)
    # Display the donut chart in the sidebar
    st.sidebar.altair_chart(make_donut(positive_percentage, 'Positive Sentiment'), use_container_width=True)

    with col1:
        st.markdown(f'<h4 style="text-align: center;">{source_name} News Titles</h4>', unsafe_allow_html=True)

        titles_per_page = 12  # Number of titles to display per page
        total_titles = len(filtered_news)
        total_pages = (total_titles // titles_per_page) + (1 if total_titles % titles_per_page > 0 else 0)

        # Page selection at the bottom of the titles
        page_number = st.selectbox("Select Page", list(range(1, total_pages + 1)), index=0)

        # Calculate the start and end index for the current page
        start_index = (page_number - 1) * titles_per_page
        end_index = start_index + titles_per_page

        # Display the titles for the selected page
        for _, row in filtered_news.iloc[start_index:end_index].iterrows():
            st.write(f"- [{row['title']}]({row['link']})")

    with col2:
    
        st.markdown(f'<h4 style="text-align: center;">{source_name} Wordcloud</h4>', unsafe_allow_html=True)
        # Create text for the word cloud from the filtered news titles
        wordcloud_text = " ".join(filtered_news['title'].tolist())
        # Generate the word cloud
        wordcloud_img = generate_wordcloud(wordcloud_text)
        # Display the word cloud image
        st.image(wordcloud_img, use_column_width=True)


        st.markdown(f'<h4 style="text-align: center;">{source_name} Industry Matching Scores</h4>', unsafe_allow_html=True)
        # Plot matching scores
        matching_scores_chart = alt.Chart(matching_scores_df).mark_bar().encode(
            x='Industry:N',
            y='Matching Score:Q',
            color=alt.Color('Matching Score:Q', scale=alt.Scale(scheme='inferno')),
            tooltip=['Industry:N', 'Matching Score:Q']
        ).properties(width=300, height=300)

        st.altair_chart(matching_scores_chart, use_container_width=True)

    # About section
    with st.expander(':information_source: :orange[About]', expanded=False):
        st.write("- :orange[**Purpose**]: This dashboard provides a comprehensive analysis of financial news from multiple sources, offering insights into the overall sentiment and industry relevance of the news headlines. It is designed to help users quickly gauge market sentiment and identify industry trends.")
        cell41, cell42 = st.columns((2, 2), gap='small')
        with cell41:
            st.write('''
                - :orange[**Features**]:
                    - **News Sentiment Analysis**: Sentiment analysis of news headlines using FinBERT, which classifies each headline as Positive, Negative, or Neutral.
                    - **Industry Matching Scores**: Scores that indicate how closely news headlines align with specific industries, based on keyword matching.
                    - **Word Cloud Visualization**: A word cloud generated from news headlines to highlight the most frequent words, giving a quick visual summary of trending topics.
            ''') 
        with cell42:
            st.write('''
                - :orange[**Visualization Tools**]:
                    - **Donut Chart**: A donut chart displaying the percentage of positive sentiment in the news, helping users quickly assess market mood.
                    - **Industry Bar Chart**: A bar chart showing the matching scores for different industries, highlighting where the news is most relevant.
                    - **Word Cloud**: A graphical representation of word frequency in the news headlines, visually emphasizing the most common themes and topics.
            ''') 
        st.write('- :orange[**Data Sources**]: News data is fetched from various sources, including Wall Street Journal, Yahoo Finance, and CNBC, through their respective RSS feeds.')
