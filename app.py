import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html
import random

# Set page configuration
st.set_page_config(
    page_title="CineMatch - Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling, animations, and 3D effects
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #24243e 50%, #302b63 100%);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styling with animated gradient */
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FF9A8B 0%, #FF6A88 55%, #FF99AC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 0.5rem;
        animation: gradientShift 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Subheader styling */
    .sub-header {
        font-size: 1.5rem;
        color: #d6d6d6;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Card styling with glassmorphism effect */
    .recommendation-card {
        background: rgba(255, 255, 255, 0.07);
        padding: 1.8rem;
        border-radius: 1.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.36);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    
    .recommendation-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* User stats styling */
    .user-stats {
        background: rgba(255, 255, 255, 0.08);
        padding: 1.8rem;
        border-radius: 1.2rem;
        margin-bottom: 1.8rem;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #FF9A8B 0%, #FF6A88 55%, #FF99AC 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 106, 136, 0.4);
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 106, 136, 0.6);
        background: linear-gradient(45deg, #FF9A8B 0%, #FF6A88 55%, #FF99AC 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #24243e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Slider styling */
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #FF9A8B 0%, #FF6A88 55%, #FF99AC 100%);
    }
    
    /* Radio button styling */
    .stRadio>div {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Checkbox styling */
    .stCheckbox>label {
        color: #d6d6d6;
        font-weight: 400;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 5px;
        margin-right: 10px;
        width: 125px;
        border-radius: 12px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Animation for cards */
    @keyframes fadeInUp {
        from { 
            opacity: 0; 
            transform: translateY(30px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    .recommendation-card, .user-stats, .metric-card {
        animation: fadeInUp 0.8s ease forwards;
    }
    
    /* Staggered animation for cards */
    .recommendation-card:nth-child(1) { animation-delay: 0.1s; }
    .recommendation-card:nth-child(2) { animation-delay: 0.2s; }
    .recommendation-card:nth-child(3) { animation-delay: 0.3s; }
    .recommendation-card:nth-child(4) { animation-delay: 0.4s; }
    .recommendation-card:nth-child(5) { animation-delay: 0.5s; }
    
    /* Pulse animation for loading */
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #FF9A8B 0%, #FF6A88 55%, #FF99AC 100%);
        border-radius: 10px;
    }
    
    /* Floating animation */
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-12px) rotate(2deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    .float {
        animation: float 4s ease-in-out infinite;
    }
    
    /* Genre tags styling */
    .genre-tag {
        display: inline-block;
        background: linear-gradient(45deg, #FF9A8B 0%, #FF6A88 55%, #FF99AC 100%);
        color: white;
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        box-shadow: 0 4px 10px rgba(255, 106, 136, 0.3);
        transition: all 0.3s ease;
    }
    
    .genre-tag:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(255, 106, 136, 0.5);
    }
    
    /* Divider styling */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 1.5rem 0;
    }
    
    /* Text highlighting */
    .highlight {
        background: linear-gradient(120deg, rgba(255,154,139,0.3) 0%, rgba(255,106,136,0.3) 100%);
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript for additional interactivity and animations
st.markdown("""
<script>
    // Add scroll effects with parallax
    window.addEventListener('scroll', function() {
        const cards = document.querySelectorAll('.recommendation-card, .metric-card');
        const header = document.querySelector('.main-header');
        const scrollY = window.scrollY;
        
        // Parallax effect for header
        if (header) {
            header.style.transform = `translateY(${scrollY * 0.4}px)`;
        }
        
        // Staggered movement for cards
        cards.forEach((card, index) => {
            const speed = 0.05 + (index * 0.03);
            card.style.transform = `translateY(${scrollY * speed}px) rotate(${scrollY * 0.001}deg)`;
        });
    });
    
    // Add mouse move effects
    document.addEventListener('mousemove', function(e) {
        const cards = document.querySelectorAll('.recommendation-card');
        const x = e.clientX / window.innerWidth;
        const y = e.clientY / window.innerHeight;
        
        cards.forEach((card, index) => {
            const moveX = (x - 0.5) * 10;
            const moveY = (y - 0.5) * 10;
            card.style.transform = `translate(${moveX}px, ${moveY}px)`;
        });
    });
</script>
""", unsafe_allow_html=True)

class StreamlitMovieRecommendationSystem:
    def __init__(self, model_path):
        """
        Initialize the recommendation system for Streamlit app
        
        Args:
            model_path (str): Path to the saved model directory
        """
        self.model_path = model_path
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.movies = None
        self.users = None
        self.ratings = None
        
    def load_model(self):
        """Load the trained model components"""
        try:
            model_components = joblib.load("recommendation_model.pkl")
            self.user_item_matrix = model_components['user_item_matrix']
            self.user_similarity_matrix = model_components['user_similarity_matrix']
            self.item_similarity_matrix = model_components['item_similarity_matrix']
            self.movies = model_components['movies']
            self.users = model_components['users']
            self.ratings = model_components['ratings']
            return True
        except FileNotFoundError:
            st.error("Model file not found: recommendation_model.pkl")
            return False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    
    def get_available_users(self):
        """Get list of available user IDs"""
        if self.user_item_matrix is None:
            return []
        return sorted(self.user_item_matrix.index.tolist())
    
    def get_user_statistics(self, user_id):
        """Get statistics for a specific user"""
        if self.ratings is None:
            return None
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        
        if len(user_ratings) == 0:
            return None
            
        stats = {
            'total_ratings': len(user_ratings),
            'avg_rating': user_ratings['rating'].mean(),
            'rating_std': user_ratings['rating'].std(),
            'favorite_genres': self._get_user_favorite_genres(user_id),
            'rating_distribution': user_ratings['rating'].value_counts().to_dict()
        }
        
        return stats
    
    def _get_user_favorite_genres(self, user_id):
        """Get user's favorite genres based on high ratings"""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        high_rated = user_ratings[user_ratings['rating'] >= 4]
        
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                     'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                     'Thriller', 'War', 'Western']
        
        genre_scores = {}
        
        for _, rating in high_rated.iterrows():
            movie_info = self.movies[self.movies['item_id'] == rating['item_id']]
            if not movie_info.empty:
                for genre in genre_cols:
                    if movie_info.iloc[0][genre] == 1:
                        genre_scores[genre] = genre_scores.get(genre, 0) + 1
        
        # Sort by frequency
        favorite_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        return favorite_genres[:5]  # Top 5 genres
    
    def collaborative_filtering_recommendations(self, user_id, k=10, n_similar_users=50):
        """
        Generate recommendations using collaborative filtering
        
        Args:
            user_id (int): Target user ID
            k (int): Number of recommendations to return
            n_similar_users (int): Number of similar users to consider
            
        Returns:
            list: Top-K movie recommendations with scores
        """
        if self.user_similarity_matrix is None or user_id not in self.user_similarity_matrix.index:
            return []
            
        # Get similar users (excluding the user themselves)
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)[1:n_similar_users+1]
        
        # Get movies the target user hasn't rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Calculate weighted ratings for unrated movies
        recommendations = {}
        
        for movie_id in unrated_movies:
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user_id, similarity_score in similar_users.items():
                if self.user_item_matrix.loc[similar_user_id, movie_id] > 0:
                    rating = self.user_item_matrix.loc[similar_user_id, movie_id]
                    weighted_sum += similarity_score * rating
                    similarity_sum += abs(similarity_score)
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations[movie_id] = predicted_rating
        
        # Sort and return top-K recommendations
        top_recommendations = sorted(recommendations.items(), 
                                   key=lambda x: x[1], reverse=True)[:k]
        
        return top_recommendations
    
    def content_based_recommendations(self, user_id, k=10):
        """
        Generate recommendations using content-based filtering
        
        Args:
            user_id (int): Target user ID
            k (int): Number of recommendations to return
            
        Returns:
            list: Top-K movie recommendations with scores
        """
        if self.user_item_matrix is None or user_id not in self.user_item_matrix.index:
            return []
            
        # Get user's rated movies and their ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) == 0:
            return []
        
        # Calculate user profile based on liked movies (rating >= 4)
        liked_movies = rated_movies[rated_movies >= 4]
        
        if len(liked_movies) == 0:
            liked_movies = rated_movies[rated_movies >= rated_movies.mean()]
        
        # Get unrated movies
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Calculate content-based scores
        recommendations = {}
        
        for movie_id in unrated_movies:
            if movie_id in self.item_similarity_matrix.index:
                # Calculate similarity to liked movies
                similarities = []
                weights = []
                
                for liked_movie_id in liked_movies.index:
                    if liked_movie_id in self.item_similarity_matrix.index:
                        sim = self.item_similarity_matrix.loc[movie_id, liked_movie_id]
                        similarities.append(sim)
                        weights.append(liked_movies[liked_movie_id])
                
                if similarities:
                    # Weighted average similarity
                    weighted_similarity = np.average(similarities, weights=weights)
                    recommendations[movie_id] = weighted_similarity
        
        # Sort and return top-K recommendations
        top_recommendations = sorted(recommendations.items(), 
                                   key=lambda x: x[1], reverse=True)[:k]
        
        return top_recommendations
    
    def hybrid_recommendations(self, user_id, k=10, cf_weight=0.7, cb_weight=0.3):
        """
        Generate hybrid recommendations combining collaborative and content-based filtering
        
        Args:
            user_id (int): Target user ID
            k (int): Number of recommendations to return
            cf_weight (float): Weight for collaborative filtering
            cb_weight (float): Weight for content-based filtering
            
        Returns:
            list: Top-K movie recommendations with combined scores
        """
        # Get recommendations from both methods
        cf_recs = dict(self.collaborative_filtering_recommendations(user_id, k=50))
        cb_recs = dict(self.content_based_recommendations(user_id, k=50))
        
        # Normalize scores to [0, 1] range
        if cf_recs:
            cf_max = max(cf_recs.values())
            cf_min = min(cf_recs.values())
            if cf_max != cf_min:
                cf_recs = {k: (v - cf_min) / (cf_max - cf_min) for k, v in cf_recs.items()}
        
        if cb_recs:
            cb_max = max(cb_recs.values())
            cb_min = min(cb_recs.values())
            if cb_max != cb_min:
                cb_recs = {k: (v - cb_min) / (cb_max - cb_min) for k, v in cb_recs.items()}
        
        # Combine scores
        all_movies = set(cf_recs.keys()) | set(cb_recs.keys())
        hybrid_scores = {}
        
        for movie_id in all_movies:
            cf_score = cf_recs.get(movie_id, 0)
            cb_score = cb_recs.get(movie_id, 0)
            hybrid_scores[movie_id] = cf_weight * cf_score + cb_weight * cb_score
        
        # Sort and return top-K recommendations
        top_recommendations = sorted(hybrid_scores.items(), 
                                   key=lambda x: x[1], reverse=True)[:k]
        
        return top_recommendations
    
    def get_movie_details(self, movie_ids):
        """Get movie details for given movie IDs"""
        if isinstance(movie_ids, (int, float)):
            movie_ids = [movie_ids]
            
        details = []
        for movie_id in movie_ids:
            movie_info = self.movies[self.movies['item_id'] == movie_id]
            if not movie_info.empty:
                details.append({
                    'item_id': movie_id,
                    'title': movie_info.iloc[0]['title'],
                    'genres': self._get_movie_genres(movie_id)
                })
        return details
    
    def _get_movie_genres(self, movie_id):
        """Get genres for a specific movie"""
        movie_info = self.movies[self.movies['item_id'] == movie_id]
        if movie_info.empty:
            return []
            
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                     'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                     'Thriller', 'War', 'Western']
        
        genres = []
        for genre in genre_cols:
            if movie_info.iloc[0][genre] == 1:
                genres.append(genre)
        return genres

def create_rating_stars(rating_score):
    """Create visual star rating based on score (expects score in [0,1])"""
    max_stars = 5
    full_stars = int(rating_score * max_stars)
    half_star = 1 if (rating_score * max_stars) - full_stars >= 0.5 else 0
    empty_stars = max_stars - full_stars - half_star
    
    stars_html = ""
    for _ in range(full_stars):
        stars_html += "⭐"
    if half_star:
        stars_html += "✨"
    for _ in range(empty_stars):
        stars_html += "☆"
    
    return stars_html

def create_genre_tags(genres):
    """Create styled genre tags"""
    if not genres:
        return ""
    
    tags_html = ""
    for genre in genres:
        tags_html += f'<span class="genre-tag">{genre}</span>'
    return tags_html


def _empty_state(title: str, body: str = ""):
    st.markdown(
        f"""
        <div style="text-align: center; margin: 2rem 0; padding: 2rem; border-radius: 16px; background: rgba(255,255,255,0.05);">
            <div class="pulse" style="font-size: 3rem;">🪄</div>
            <h3 style="color: #d6d6d6; margin-top: 1rem;">{title}</h3>
            <p style="color: #a8a8a8;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Main function to run the Streamlit app"""
    
    # Initialize the recommendation system
    #model_path = st.secrets.get("model_path", 'D:/Internship/task_5/model') if hasattr(st, 'secrets') else 'D:/Internship/task_5/model'
    #model_path = getattr(st, "secrets", {}).get("model_path", "model.pkl")
    # Load model path safely
    model_path = "recommendation_model.pkl" # default path in your project folder
    if os.path.exists(model_path):
        recommender = joblib.load(model_path)
    else:
        st.error(f"Model file not found at {model_path}. Please check the path.")
        st.stop()


    recommender = StreamlitMovieRecommendationSystem(model_path)
    
    # App title and description
    st.markdown('<h1 class="main-header float">🎬 CineMatch</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Discover Your Next Favorite Movie</h3>', unsafe_allow_html=True)
    
    # Decorative elements
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div class="pulse" style="font-size: 4rem;">✨🎥🍿✨</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load the model with a nice animation
    with st.spinner(""):
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div style="text-align: center; margin: 2rem 0; padding: 2rem; border-radius: 16px; background: rgba(255,255,255,0.05);">
            <div class="pulse" style="font-size: 3rem;">🍿</div>
            <h3 style="color: #d6d6d6; margin-top: 1rem;">Initializing Recommendation Engine</h3>
            <p style="color: #a8a8a8;">Loading movie database and algorithms...</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not recommender.load_model():
            st.error("Failed to load the recommendation model. Please check the model path or file.")
            _empty_state("Model not loaded", "Provide a valid path in st.secrets['model_path'] or update the hardcoded path.")
            return
        
        # Simulate loading for better UX
        time.sleep(1.2)
        loading_placeholder.empty()
    
    # Sidebar for user input with modern styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
            <h2 style="color: white; margin-top: 0; font-weight: 600;">User Selection</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Get available users
        available_users = recommender.get_available_users()
        if not available_users:
            st.warning("No users found in the model. Check the data inside the pickle file.")
            return
        
        # User selection
        default_user_index = available_users.index(196) if 196 in available_users else 0
        selected_user = st.selectbox(
            "Select a User ID:",
            options=available_users,
            index=default_user_index
        )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Recommendation method selection
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚙️</div>
            <h3 style="color: white; margin-top: 0; font-weight: 600;">Recommendation Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        method = st.radio(
            "Select Recommendation Method:",
            ["Hybrid", "Collaborative Filtering", "Content-Based Filtering"],
            index=0,
        )
        
        # Number of recommendations
        num_recommendations = st.slider(
            "Number of Recommendations:",
            min_value=5,
            max_value=20,
            value=10,
        )
        
        # Hybrid weights (only show if hybrid method is selected)
        if method == "Hybrid":
            cf_weight = st.slider(
                "Collaborative Filtering Weight:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
            )
            cb_weight = 1.0 - cf_weight
            st.markdown(f"<p style='color: #d6d6d6;'>Content-Based Filtering Weight: <span class='highlight'>{cb_weight:.1f}</span></p>", unsafe_allow_html=True)
        else:
            cf_weight, cb_weight = 0.7, 0.3  # defaults, unused for non-hybrid
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Generate recommendations button
        generate_btn = st.button("🎬 Generate Recommendations", type="primary", use_container_width=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Additional options
        show_history = st.checkbox("Show User's Rating History")
        show_distributions = st.checkbox("Show Rating Distributions")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # System information
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <h3 style="color: white; margin-top: 0; font-weight: 600;">System Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<p>👥 Total Users: <span class='highlight'>{len(recommender.users)}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p>🎬 Total Movies: <span class='highlight'>{len(recommender.movies)}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p>⭐ Total Ratings: <span class='highlight'>{len(recommender.ratings)}</span></p>", unsafe_allow_html=True)
        
        # Data sparsity calculation
        total_possible_ratings = max(1, len(recommender.users) * len(recommender.movies))
        actual_ratings = len(recommender.ratings)
        sparsity = 1 - (actual_ratings / total_possible_ratings)
        st.markdown(f"<p>📈 Data Sparsity: <span class='highlight'>{sparsity:.4f}</span></p>", unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Get user statistics
        user_stats = recommender.get_user_statistics(selected_user)
        
        # Display user statistics
        if user_stats:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">👤</div>
                <h2 style="color: white; margin-top: 0; font-weight: 600;">User Profile</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="user-stats">', unsafe_allow_html=True)
            
            # Metrics in a grid
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.markdown(f'<div class="metric-card"><h4 style="margin: 0; color: #d6d6d6;">Total Ratings</h4><h2 style="margin: 0; color: #FF9A8B;">{user_stats["total_ratings"]}</h2></div>', unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f'<div class="metric-card"><h4 style="margin: 0; color: #d6d6d6;">Avg Rating</h4><h2 style="margin: 0; color: #FF9A8B;">{user_stats["avg_rating"]:.2f}</h2></div>', unsafe_allow_html=True)
            
            # Favorite genres
            favorite_genres = [genre for genre, count in user_stats['favorite_genres']]
            if favorite_genres:
                st.markdown("<h4 style='color: #d6d6d6; margin-bottom: 0.5rem;'>Favorite Genres:</h4>", unsafe_allow_html=True)
                genres_html = create_genre_tags(favorite_genres)
                st.markdown(genres_html, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            _empty_state("No profile stats yet", "This user has no ratings. Try selecting another user.")
    
    with col2:
        if generate_btn:
            with st.spinner(""):
                # Animation while processing
                processing_placeholder = st.empty()
                processing_placeholder.markdown(f"""
                <div style="text-align: center; margin: 2rem 0; padding: 2rem; border-radius: 16px; background: rgba(255,255,255,0.05);">
                    <div class="pulse" style="font-size: 3rem;">{"🎥" if method == "Content-Based Filtering" else "👥" if method == "Collaborative Filtering" else "⚡"}</div>
                    <h3 style="color: #d6d6d6; margin-top: 1rem;">Analyzing Preferences</h3>
                    <p style="color: #a8a8a8;">Using {method} to find your perfect matches...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Simulate processing for better UX
                time.sleep(1.2)
                
                # Get recommendations based on selected method
                if method == "Collaborative Filtering":
                    recommendations = recommender.collaborative_filtering_recommendations(
                        selected_user, k=num_recommendations
                    )
                elif method == "Content-Based Filtering":
                    recommendations = recommender.content_based_recommendations(
                        selected_user, k=num_recommendations
                    )
                else:  # Hybrid
                    recommendations = recommender.hybrid_recommendations(
                        selected_user, k=num_recommendations, 
                        cf_weight=cf_weight, cb_weight=cb_weight
                    )
                
                processing_placeholder.empty()
                
                # Display recommendations
                if recommendations:
                    st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 2rem;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎭</div>
                        <h2 style="color: white; margin-top: 0; font-weight: 600;">Top {num_recommendations} Recommendations</h2>
                        <p style="color: #a8d8ea;">Method: {method}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, (movie_id, score) in enumerate(recommendations, 1):
                        movie_details = recommender.get_movie_details([movie_id])
                        if movie_details:
                            movie = movie_details[0]
                            genres = movie['genres']
                            # For hybrid path, scores normalized to [0,1]; others may not be. Clamp to [0,1] for stars only.
                            star_score = float(np.clip(score, 0.0, 1.0))
                            stars = create_rating_stars(star_score)
                            genre_tags = create_genre_tags(genres)
                            
                            with st.container():
                                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                                
                                # Create columns for layout
                                col_a, col_b = st.columns([3, 1])
                                
                                with col_a:
                                    st.markdown(f"<h3 style='margin: 0; color: white;'>{i}. {movie['title']}</h3>", unsafe_allow_html=True)
                                    if genre_tags:
                                        st.markdown(f"<div style='margin-top: 0.5rem;'>{genre_tags}</div>", unsafe_allow_html=True)
                                
                                with col_b:
                                    st.markdown(f"<p style='margin: 0; color: #FF9A8B; font-weight: 600;'>Score: {score:.3f}</p>", unsafe_allow_html=True)
                                    st.markdown(f"<div style='font-size: 1.2rem;'>{stars}</div>", unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No recommendations available for this user.")
        
        # Display user's rating history if requested
        if show_history:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📝</div>
                <h2 style="color: white; margin-top: 0; font-weight: 600;">User Rating History</h2>
            </div>
            """, unsafe_allow_html=True)
            
            user_ratings = recommender.ratings[recommender.ratings['user_id'] == selected_user].copy()
            if user_ratings.empty:
                _empty_state("No ratings found for this user.")
            else:
                user_ratings_with_titles = user_ratings.merge(
                    recommender.movies[['item_id', 'title']], on='item_id'
                )[['title', 'rating', 'timestamp']]
                
                # Convert timestamp to readable date
                try:
                    user_ratings_with_titles['date'] = pd.to_datetime(
                        user_ratings_with_titles['timestamp'], unit='s'
                    )
                except Exception:
                    # If timestamp isn't epoch seconds, try direct parse
                    user_ratings_with_titles['date'] = pd.to_datetime(
                        user_ratings_with_titles['timestamp'], errors='coerce'
                    )
                
                user_ratings_with_titles.sort_values('date', ascending=False, inplace=True)
                
                # Show table
                st.dataframe(
                    user_ratings_with_titles.rename(columns={"title": "Movie", "rating": "Rating", "date": "Date"})[
                        ["Movie", "Rating", "Date"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                
                # Download
                csv = user_ratings_with_titles.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Rating History (CSV)",
                    data=csv,
                    file_name=f"user_{selected_user}_ratings.csv",
                    mime='text/csv',
                    use_container_width=True,
                )
        
        if show_distributions:
            # Rating distribution for the selected user
            user_ratings = recommender.ratings[recommender.ratings['user_id'] == selected_user]
            if not user_ratings.empty:
                dist = user_ratings['rating'].value_counts().sort_index().reset_index()
                dist.columns = ['Rating', 'Count']
                bar_fig = px.bar(dist, x='Rating', y='Count', title='User Rating Distribution')
                bar_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(bar_fig, use_container_width=True)
            
            # Global popular genres (quick view)
            genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                          'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                          'Thriller', 'War', 'Western']
            if set(genre_cols).issubset(set(recommender.movies.columns)):
                genre_sums = recommender.movies[genre_cols].sum().sort_values(ascending=False).head(10)
                genre_df = genre_sums.reset_index()
                genre_df.columns = ['Genre', 'Count']
                pie_fig = px.pie(genre_df, values='Count', names='Genre', title='Top Genres in Catalog')
                pie_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(pie_fig, use_container_width=True)
    
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center; color:#a8a8a8; font-size:0.9rem; padding-bottom:2rem;">
            Built with ❤️ using Streamlit • CineMatch © 2025
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
