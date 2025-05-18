import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 26px;
        font-weight: bold;
        color: #FF8C00;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .card {
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .highlight {
        color: #FF4B4B;
        font-weight: bold;
    }
    .film-title {
        font-size: 18px;
        font-weight: bold;
        color: #343a40;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
    .metric-value {
        font-size: 18px;
        font-weight: bold;
        color: #343a40;
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

# Create feature for recommendation
@st.cache_data
def create_feature_matrix(df):
    features = []
    
    # Determine which columns to use as features
    if 'Description' in df.columns or any('description' in col.lower() for col in df.columns):
        desc_col = next(col for col in df.columns if 'description' in col.lower())
        features.append(df[desc_col].fillna('').astype(str))
    
    if 'Genre' in df.columns or any('genre' in col.lower() for col in df.columns):
        genre_col = next(col for col in df.columns if 'genre' in col.lower())
        features.append(df[genre_col].fillna('').astype(str))
    
    if 'Director' in df.columns or any('director' in col.lower() for col in df.columns):
        dir_col = next(col for col in df.columns if 'director' in col.lower())
        features.append(df[dir_col].fillna('').astype(str))
    
    # Combine all features
    if features:
        combined_features = features[0]
        for i in range(1, len(features)):
            combined_features += ' ' + features[i]
    else:
        # If no suitable text features found, use the movie title
        combined_features = df[movie_title_col].fillna('').astype(str)
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined_features)
    
    return tfidf_matrix

# Function to get movie recommendations
@st.cache_data
def get_recommendations(movie_title, _df, tfidf_matrix, top_n=5):
    # Get the index of the movie that matches the title
    idx = _df[_df[movie_title_col] == movie_title].index[0]
    
    # Compute the cosine similarity matrix
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix)[0]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top_n most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top top_n most similar movies with their similarity scores
    recommendations = _df.iloc[movie_indices].copy()
    recommendations['Similarity'] = [i[1] for i in sim_scores]
    
    return recommendations

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
    # Create feature matrix for recommendations
    tfidf_matrix = create_feature_matrix(df)
    
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
                                
                        # Get recommendations button
                        if st.button(f"Get similar movies to {movie[movie_title_col]}", key=f"recommend_{idx}"):
                            st.session_state.selected_movie = movie[movie_title_col]
                            st.session_state.show_recommendations = True

    st.markdown("---")

    # Show recommendations if requested
    if 'show_recommendations' in st.session_state and st.session_state.show_recommendations:
        st.markdown("<h2 class='sub-header'>Recommended Movies</h2>", unsafe_allow_html=True)
        recommendations = get_recommendations(st.session_state.selected_movie, df, tfidf_matrix, top_n=6)
        
        st.markdown(f"<h3>Movies similar to <span class='highlight'>{st.session_state.selected_movie}</span></h3>", unsafe_allow_html=True)
        
        # Create rows of recommendations
        for i in range(0, len(recommendations), movies_per_row):
            cols = st.columns(movies_per_row)
            
            for j in range(movies_per_row):
                idx = i + j
                if idx < len(recommendations):
                    movie = recommendations.iloc[idx]
                    
                    with cols[j]:
                        with st.container():
                            st.markdown(f"<div class='card'><p class='film-title'>{movie[movie_title_col]} ({movie.get('Year of Release', '')})</p></div>", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"<p class='metric-label'>Rating</p><p class='metric-value'>⭐ {movie.get('Movie Rating', 'N/A')}</p>", unsafe_allow_html=True)
                            with col2:
                                similarity = movie['Similarity'] * 100
                                st.markdown(f"<p class='metric-label'>Match</p><p class='metric-value'>{similarity:.1f}%</p>", unsafe_allow_html=True)

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
<div style="text-align: center">
    <p>Created with ❤️ using Streamlit and Python</p>
    <p>Data source: IMDb Top 1000 Movies</p>
</div>
""", unsafe_allow_html=True)
