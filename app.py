import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# Set page config
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* General Styles */
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .main-header:hover {
        color: #FF6B6B;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
    }
    .sub-header {
        font-size: 26px;
        font-weight: bold;
        color: #FF8C00;
        margin-top: 20px;
        margin-bottom: 10px;
        transition: color 0.3s ease;
    }
    .sub-header:hover {
        color: #FFA040;
    }
    
    /* Card Styles */
    .card {
        border-radius: 12px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        border-color: #FF4B4B;
    }
    
    /* Text Styles */
    .highlight {
        color: #FF4B4B;
        font-weight: bold;
        transition: color 0.3s ease;
    }
    .highlight:hover {
        color: #FF6B6B;
    }
    .film-title {
        font-size: 18px;
        font-weight: bold;
        color: #343a40;
        transition: all 0.3s ease;
    }
    .film-title:hover {
        color: #FF4B4B;
    }
    
    /* Metrics Styles */
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        transition: color 0.3s ease;
    }
    .metric-value {
        font-size: 18px;
        font-weight: bold;
        color: #343a40;
        transition: all 0.3s ease;
    }
    .metric-value:hover {
        color: #FF4B4B;
        transform: scale(1.05);
    }
    
    /* Button Styles */
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .streamlit-expanderHeader:hover {
        background-color: #FF4B4B15;
    }
    
    /* Footer Styles */
    .footer {
        text-align: center;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-top: 30px;
    }
    .footer a {
        color: #FF4B4B;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    .footer a:hover {
        color: #FF6B6B;
        text-decoration: underline;
    }
    .footer-text {
        font-size: 14px;
        color: #6c757d;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    # Clean up column names
    df.columns = df.columns.str.strip()
    # Get meaningful column name for movie title
    if 'Movie Name' in df.columns:
        movie_title_col = 'Movie Name'
    else:
        # Find the column likely to contain movie titles
        for col in df.columns:
            if 'movie' in col.lower() or 'title' in col.lower() or 'name' in col.lower():
                movie_title_col = col
                break
        else:
            # If no suitable column found, use the first column after index
            movie_title_col = df.columns[1]  # Often the second column contains movie names
    
    return df, movie_title_col

# Load data
with st.spinner('Loading movie data...'):
    df, movie_title_col = load_data()
    # Get list of all genres
    if 'Genre' in df.columns or any('genre' in col.lower() for col in df.columns):
        genre_col = next(col for col in df.columns if 'genre' in col.lower())
        # Extract all unique genres
        all_genres = []
        for genres in df[genre_col].dropna():
            genre_list = [g.strip() for g in re.split(r'[,|/]', genres)]
            all_genres.extend(genre_list)
        unique_genres = sorted(list(set(all_genres)))
    else:
        unique_genres = []

# Create a sidebar for filters and search
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Movie Filters</h2>", unsafe_allow_html=True)
    
    # Search box
    search_term = st.text_input("🔍 Search by movie title", "")
      # Year range filter
    if 'Year of Release' in df.columns or any('year' in col.lower() for col in df.columns):
        year_col = next(col for col in df.columns if 'year' in col.lower())
        # Extract only numerical years using regex
        df['year_numeric'] = df[year_col].astype(str).str.extract(r'(\d{4})').astype(float)
        min_year, max_year = int(df['year_numeric'].min()), int(df['year_numeric'].max())
        year_range = st.slider("📅 Year of Release", min_year, max_year, (min_year, max_year))
    else:
        year_range = None
    
    # Rating filter
    if 'Movie Rating' in df.columns or any('rating' in col.lower() for col in df.columns):
        rating_col = next(col for col in df.columns if 'rating' in col.lower())
        min_rating = float(df[rating_col].min())
        max_rating = float(df[rating_col].max())
        rating_filter = st.slider("⭐ Minimum Rating", min_rating, max_rating, min_rating, 0.1)
    else:
        rating_filter = None
    
    # Genre filter
    if unique_genres:
        selected_genres = st.multiselect("🎭 Genres", unique_genres)
    else:
        selected_genres = []
    
    # Director filter
    if 'Director' in df.columns or any('director' in col.lower() for col in df.columns):
        director_col = next(col for col in df.columns if 'director' in col.lower())
        directors = sorted(df[director_col].dropna().unique().tolist())
        selected_director = st.selectbox("🎬 Director", ["All"] + directors)
    else:
        selected_director = "All"
    
    # Reset filters button
    if st.button("🔄 Reset Filters"):
        search_term = ""
        if year_range:
            year_range = (min_year, max_year)
        if rating_filter is not None:
            rating_filter = min_rating
        selected_genres = []
        selected_director = "All"

# Apply filters to the DataFrame

# Apply filters to the DataFrame
filtered_df = df.copy()

# Apply text search
if search_term:
    filtered_df = filtered_df[filtered_df[movie_title_col].str.contains(search_term, case=False)]

# Apply year filter
if year_range:
    filtered_df = filtered_df[
        (filtered_df['year_numeric'] >= year_range[0]) & 
        (filtered_df['year_numeric'] <= year_range[1])
    ]

# Apply rating filter
if rating_filter is not None:
    rating_col = next(col for col in df.columns if 'rating' in col.lower())
    filtered_df = filtered_df[filtered_df[rating_col] >= rating_filter]

# Apply genre filter
if selected_genres:
    genre_col = next(col for col in df.columns if 'genre' in col.lower())
    genre_filter = filtered_df[genre_col].apply(
        lambda x: any(genre in str(x) for genre in selected_genres) if pd.notna(x) else False
    )
    filtered_df = filtered_df[genre_filter]

# Apply director filter
if selected_director != "All":
    director_col = next(col for col in df.columns if 'director' in col.lower())
    filtered_df = filtered_df[filtered_df[director_col] == selected_director]

# Main content - show movies in the filtered DataFrame
st.markdown("<h2 class='sub-header'>Movies Catalog</h2>", unsafe_allow_html=True)

if len(filtered_df) == 0:
    st.warning("No movies found with the current filters. Try adjusting your search criteria.")
else:
    # Display movie catalog
    movies_per_row = 3
    
    # Create rows of movies
    for i in range(0, len(filtered_df), movies_per_row):
        cols = st.columns(movies_per_row)
        
        for j in range(movies_per_row):
            idx = i + j
            if idx < len(filtered_df):
                movie = filtered_df.iloc[idx]
                
                with cols[j]:
                    with st.expander(f"{movie[movie_title_col]} ({movie.get('Year of Release', '')})"):
                        st.write(f"**Rating:** ⭐ {movie.get('Movie Rating', 'N/A')}")
                        
                        if 'Director' in movie:
                            st.write(f"**Director:** {movie['Director']}")
                        elif any('director' in col.lower() for col in movie.index):
                            director_col = next(col for col in movie.index if 'director' in col.lower())
                            st.write(f"**Director:** {movie[director_col]}")
                        
                        if 'Genre' in movie:
                            st.write(f"**Genre:** {movie['Genre']}")
                        elif any('genre' in col.lower() for col in movie.index):
                            genre_col = next(col for col in movie.index if 'genre' in col.lower())
                            st.write(f"**Genre:** {movie[genre_col]}")
                        
                        if 'Watch Time' in movie:
                            st.write(f"**Duration:** {movie['Watch Time']} min")
                        elif any('time' in col.lower() for col in movie.index) or any('duration' in col.lower() for col in movie.index):
                            time_col = next((col for col in movie.index if 'time' in col.lower() or 'duration' in col.lower()), None)
                            if time_col:
                                st.write(f"**Duration:** {movie[time_col]} min")
                                
                        if 'Description' in movie:
                            st.write(f"**Plot:** {movie['Description']}")
                        elif any('description' in col.lower() for col in movie.index) or any('plot' in col.lower() for col in movie.index):
                            desc_col = next((col for col in movie.index if 'description' in col.lower() or 'plot' in col.lower()), None)
                            if desc_col:
                                st.write(f"**Plot:** {movie[desc_col]}")

    st.markdown("---")

# Data insights section
with st.expander("📊 Movie Insights"):
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Rating Distribution", "Year Distribution", "Top Directors"])
    with tab1:
        # Rating distribution
        if 'Movie Rating' in df.columns or any('rating' in col.lower() for col in df.columns):
            rating_col = next(col for col in df.columns if 'rating' in col.lower())
            fig = px.histogram(df, x=rating_col, nbins=20, title="Movie Rating Distribution",
                           labels={rating_col: 'Rating', 'count': 'Number of Movies'},
                           color_discrete_sequence=['#FF4B4B'])
            fig.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rating data not available for visualization.")
    
    with tab2:
        # Year distribution
        if 'year_numeric' in df.columns:
            fig = px.histogram(df, x='year_numeric', title="Movies by Year",
                           labels={'year_numeric': 'Year of Release', 'count': 'Number of Movies'},
                           color_discrete_sequence=['#FF8C00'])
            fig.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Year data not available for visualization.")
    
    with tab3:
        # Top directors by average rating
        if ('Director' in df.columns or any('director' in col.lower() for col in df.columns)) and \
           ('Movie Rating' in df.columns or any('rating' in col.lower() for col in df.columns)):
            director_col = next(col for col in df.columns if 'director' in col.lower())
            rating_col = next(col for col in df.columns if 'rating' in col.lower())
            
            top_directors = df.groupby(director_col)[rating_col].agg(['mean', 'count']).reset_index()
            top_directors = top_directors[top_directors['count'] >= 2]  # At least 2 movies
            top_directors = top_directors.sort_values('mean', ascending=False).head(10)
            
            fig = px.bar(top_directors, x=director_col, y='mean', 
                      title="Top Directors by Average Rating (min 2 movies)",
                      labels={director_col: 'Director', 'mean': 'Average Rating'},
                      color='count', color_continuous_scale='Viridis',
                      text_auto='.2f')
            fig.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Director or rating data not available for visualization.")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p class="footer-text">Created with ❤️ by <a href="https://sahilfolio.live/" target="_blank">Sahil</a></p>
    <p class="footer-text">Built using Streamlit and Python</p>
    <p class="footer-text">Data source: IMDb Top 1000 Movies</p>
    <p class="footer-text">
        <a href="https://sahilfolio.live/" target="_blank">Portfolio</a> • 
        <a href="https://github.com/Sahilll94" target="_blank">GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
