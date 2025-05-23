﻿# Movie Recommendation System

A content-based movie recommendation system built with Streamlit that suggests similar movies based on descriptions, genres, and directors using the IMDb Top 1000 Movies dataset.

## Features

- **Content-Based Recommendations**: Get movie recommendations based on similarity in plot, genre, and director
- **Advanced Filtering**: Filter by title, year, rating, genre, and director
- **Interactive UI**: Expandable movie details and intuitive interface
- **Data Visualizations**: Explore rating distributions, year trends, and top directors

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Steps to Run

1. **Clone the repository**
   ```powershell
   git clone https://github.com/yourusername/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
   ```

2. **Create a virtual environment (optional but recommended)**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```

3. **Install the required packages**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```powershell
   streamlit run app.py
   ```

5. **Access the app in your browser**
   - The app will automatically open in your default browser, or you can access it at http://localhost:8501

## Dataset

The dataset includes the top 1000 movies as rated on IMDb, providing details such as:
- Movie title
- Year of release
- Genre
- Director
- IMDb rating
- Duration
- Plot description

## How It Works

1. **TF-IDF Vectorization**: Converts movie descriptions, genres, and directors into numerical vectors
2. **Cosine Similarity**: Measures the similarity between different movies
3. **Content-Based Filtering**: Recommends movies with similar content features

## Troubleshooting

- **Missing dependencies**: Ensure all packages in requirements.txt are installed
- **Year format issues**: The system automatically handles mixed year formats (e.g., "III 2018")
- **Browser compatibility**: Works best with Chrome, Firefox, or Edge

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data source: IMDb Top 1000 Movies
- Built with Streamlit, Pandas, Scikit-learn, and Plotly
