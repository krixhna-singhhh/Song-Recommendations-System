Student Details

Name: Krishnapal Rajput

Registration Number: 25MIM10084

Course: Fundamentals in AI and ML

Course Code: CSA2001

Song Recommendation System

Overview

This is a simple AI/ML-based Song Recommendation System made for the Fundamentals in AI and ML course. It recommends songs by comparing their musical features with the features of a song selected by the user.

The project uses danceability, energy, valence, tempo, and acousticness as the main features. It provides two recommendation methods: Cosine Similarity and K-Nearest Neighbors (KNN).

The application runs from the command line, so it does not need a separate graphical setup.

Student Details

Name: Krishnapal Rajput
Registration Number: 25MIM10084
Course: Fundamentals in AI and ML
Course Code: CSA2001

Features

1. List the songs available in the dataset.
2. Recommend similar songs using Cosine Similarity.
3. Recommend similar songs using KNN.
4. Evaluate recommendations using Precision@K.
5. Handle invalid song names and command-line input.
6. Run automated tests with Pytest.

Technologies Used

Python
Pandas
NumPy
Scikit-learn
Pytest
CSV

Installation

1. Install Python 3.9 or later.
2. Open a terminal inside the project folder.
3. Install the required libraries:

pip install -r requirements.txt

How to Run

List all songs:

python main.py list

Get recommendations with Cosine Similarity:

python main.py recommend --song "Neon Nights" --top 5 --method cosine

Get recommendations with KNN:

python main.py recommend --song "Neon Nights" --top 5 --method knn

Evaluate the system:

python main.py evaluate

Run Tests

pytest -q

Project Structure

song-recommendation-system/
├── README.md
├── statement.md
├── main.py
├── requirements.txt
├── .gitignore
├── data/
│   └── songs.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── recommender.py
│   ├── evaluator.py
│   └── cli.py
├── tests/
│   ├── __init__.py
│   └── test_recommender.py
└── docs/
    ├── diagrams/
    │   ├── architecture.png
    │   ├── workflow.png
    │   ├── use_case.png
    │   ├── sequence.png
    │   ├── component.png
    │   └── er_schema.png
    └── screenshots/
        ├── recommendation_output.png
        ├── evaluation_output.png
        └── test_output.png

Dataset

The included songs.csv is a small synthetic educational dataset containing 30 songs from five genres. It contains song details and numerical audio-style features. The dataset is included directly in the project so that the application can run without an external API.

How the Recommendation Works

First, the numerical features are standardized using StandardScaler. The system then compares songs using Cosine Similarity or finds nearby songs using KNN with cosine distance. The selected songs are returned as recommendations.

Evaluation

Precision@K is used as a simple evaluation measure. For this educational dataset, songs from the same genre are treated as relevant recommendations.

Notes

The project is designed to be easy to run from a normal terminal and easy to extend with a larger dataset or a web interface in the future.
