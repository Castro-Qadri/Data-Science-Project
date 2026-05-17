# Movie Recommendation with Sentiment Analysis System

This project combines content-based movie recommendation with review sentiment analysis. The notebook builds the recommendation workflow, performs EDA, trains and compares sentiment models, and evaluates the results. The Streamlit app lets users search for a movie, inspect movie details, read reviews, and see predicted review sentiment.

## Dataset

- TMDB 5000 Movie Dataset from Kaggle: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- Movie posters and review metadata are fetched from the TMDB API
- The IMDB review dataset is required for the sentiment-analysis notebook sections

## Project Structure

- [Recommendation_System.ipynb](Recommendation_System.ipynb): notebook for preprocessing, EDA, recommendation logic, sentiment training, and evaluation
- [WebApp.py](WebApp.py): Streamlit interface for recommendations and sentiment display
- [Model/](Model): saved recommendation and sentiment artifacts
- [.streamlit/config.toml](.streamlit/config.toml): Streamlit theme configuration

## Features

- Content-based recommendations from movie metadata
- Movie poster, cast, genre, release date, and rating display
- Sentiment classification of reviews as positive or negative
- Model comparison and evaluation for sentiment analysis
- A cleaned dark cinematic UI in Streamlit

## Setup

1. Create and activate the project environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Make sure the TMDB API key is available in your environment as `API_KEY`.

## Run the App

```bash
streamlit run WebApp.py
```

## Notebook Notes

- The recommendation code uses sparse TF-IDF and cosine-style similarity to avoid large dense matrices.
- The IMDB dataset must be present before running the sentiment-analysis cells.
- The notebook includes EDA plots, sentiment-model comparison, and evaluation plots for the final report.

## Dependencies

- Streamlit
- joblib
- pandas
- requests
- scikit-learn
- matplotlib
- seaborn
- nltk

## Team

Created by Group No 3

Members: Muhammad Ahmad Ijaz, Abubakar Amir, Muhammad Umer

Course: Data Science Semester Project 2026

Repository: https://github.com/Castro-Qadri/Data-Science-Project
