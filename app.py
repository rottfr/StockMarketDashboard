import streamlit as st

# Set the page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation with custom bold style
st.sidebar.markdown("<h2 style='font-weight: bold;'>Navigation</h2>", unsafe_allow_html=True)

# Define a dictionary to map page names to their corresponding functions
pages = {
    "Welcome!": "modules.page0",
    "Market Overview": "modules.page1",
    "Financial News": "modules.page2",
    "Stock Evaluation": "modules.page3",
    "Sector Analysis": "modules.page4"
}

# Sidebar image logic (only show if not on the Welcome page)
selected_page_title = st.sidebar.radio("", list(pages.keys()), format_func=lambda x: f"**{x}**")

# Map the page titles to the corresponding image paths
page_images = {
    "Market Overview": "Images/stocks.jpg",
    "Financial News": "Images/stocks3.jpg",
    "Stock Evaluation": "Images/stocks2.jpeg",
    "Sector Analysis": "Images/stocks1.jpg"
}

# Display the corresponding image in the sidebar if not on the Welcome page
if selected_page_title != "Welcome!":
    image_path = page_images.get(selected_page_title)
    if image_path:
        st.sidebar.image(image_path, use_column_width=True)

# Import the selected module dynamically
selected_page = pages[selected_page_title]
module = __import__(selected_page, fromlist=['show_page'])
module.show_page()
