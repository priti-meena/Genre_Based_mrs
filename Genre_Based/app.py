import streamlit as st
import json
from Classifier import KNearestNeighbours

from operator import itemgetter
import requests
from PIL import Image
from streamlit_lottie import st_lottie

#Setting the Page Configuration
img = Image.open('movie.png')
st.set_page_config(page_title='GenreBased' , page_icon=img , layout="centered",initial_sidebar_state="expanded")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_start = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_CTaizi.json")

# Load data and movies list from corresponding JSON files
with open(r'data.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)
with open(r'titles.json', 'r+', encoding='utf-8') as f:
    movie_titles = json.load(f)

def knn(test_point, k):
    # Create dummy target variable for the KNN Classifier
    target = [0 for item in movie_titles]
    # Instantiate object for the Classifier
    model = KNearestNeighbours(data, target, test_point, k=k)
    # Run the algorithm
    model.fit()
    # Distances to most distant movie
    max_dist = sorted(model.distances, key=itemgetter(0))[-1]
    # Print list of 10 recommendations < Change value of k for a different number >
    table = list()
    for i in model.indices:
        # Returns back movie title and imdb link
        table.append([movie_titles[i][0], movie_titles[i][2]])
    return table

if __name__ == '__main__':
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
              'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
              'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    
    movies = [title[0] for title in movie_titles]

    with st.container():
     left_column, right_column = st.columns(2)
     with left_column:
         st.write("")
         st.title('MOVIE RECOMMENDER SYSTEM') 
     with right_column:
         st_lottie(lottie_start, height=300,width=400, key="start")



options = st.multiselect('SELECT GENRE(S):', genres)
if options:
            imdb_score = st.slider('IMDb score:', 1, 10, 8)
            n = st.number_input('Number of movies:', min_value=1, max_value=20, step=1)
            st.write("*To learn more about the movie, click on the IMDb link.*")

            test_point = [1 if genre in options else 0 for genre in genres]
            test_point.append(imdb_score)
            table = knn(test_point, n)
            st.write("")
            st.write("")
            st. markdown("<h1 style='text-align: center; color:#F7D7D1;'>| RECOMMENDED MOVIES |</h1>", unsafe_allow_html=True)
            st.write("")
            st.write("")
            
            for movie, link in table:
                # Displays movie title with link to imdb
                st.info(movie)
                st.markdown(f" 🟤 IMDb Link - [{movie}]({link})")

else:
                st.write("*You Can Change The IMDb Rating And The Number Of Movies*")
                        

