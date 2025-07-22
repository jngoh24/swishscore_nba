# SwishScore - NBA Shot Outcome Prediction Model (xP)

## Overview
This project builds a machine learning model to predict whether an NBA shot will be made or missed. Based on this prediction, it calculates a new statistic called "Expected Points" (xP) for each shot. Inspired by the xG statistic in soccer, the model uses various features extracted from historical NBA data to predict the likelihood of a successful shot.

## Data Sources
The dataset consists of data from the **2021-2022 NBA season**, sourced from three different locations:
1. **Shots Dataset** – Contains shot attempt details, including location, shot type, outcome, etc. (https://github.com/DomSamangy/NBA_Shots_04_24/blob/main/NBA_2022_Shots.csv.zip)
2. **Player Dataset** – Includes player-specific attributes such as position, ppg, fg%, ortg, etc.
              (https://www.nbastuffer.com/2021-2022-nba-player-stats/)
4. **Team Dataset** – Features team-level statistics, including pace, offensive rating, and defensive efficiency, etc. (https://www.nbastuffer.com/2021-2022-nba-team-stats/)

## Methodology

### My NBA Data Explorer
Check out the live Streamlit app here:  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://[your-streamlit-cloud-app-url](https://swishscorenba-kwlmtqwpoajxu4rqqpcvzu.streamlit.app/))

#### swish_eda
  - **Data Cleaning & Preprocessing:** Merging datasets, handling missing values, and feature engineering.
  - **Feature Engineering:** Creating relevant shot prediction features (e.g., in game fg%, last shot outcome, etc.).
  - **Exploratory Data Analysis:** Analyzing distributions, correlations, and key insights from the data to to select relevant features.
 #### xP_model
  - **Model Selection:** Implementing and comparing multiple machine learning models (e.g., logistic regression, random forests, and deep      learning models).
  - **xP Creation:** Using the model's propability of a made shot, xP is the product of shot_pts (2 or 3) multiplied by xP.
  - **xP Evaluation:** Evaluated each team and their performance in regards to xP - whether they **Outperformed** (scored *more* points than   expected) or **Underperformed** (scored less points than expected)
  - **Evaluation Metrics:** Assessing model performance using accuracy, precision, recall, confusion matrix, and ROC-AUC.


## Insights
- OKC, WAS, PHO, DAL, BOS outperfromed their xP most frequently, all above 71%
- ORL, TOR, CHI, NYK, DET underperformed their xP most frequently, all above 52%
  

## Future Improvements
- A major data component missing from this model is defender data, specifically defender location data. Intuitively, understanding who the closest defender is and how far away from the shot taker they are will greatly improve this model.
- Incorporate real-time shot tracking data.
- Improve model interpretability with SHAP values.
- Experiment with advanced deep learning architectures.
