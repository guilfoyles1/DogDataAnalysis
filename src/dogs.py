import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_echarts import st_echarts
import statsmodels.api as sm
from scipy.stats import pearsonr
import numpy as np

# Load dataset
df = pd.read_csv('/Users/shaynaguilfoyle/Final349/DogDataAnalysis/Dog Breads Around The World.csv')

# Sidebar filters
st.sidebar.header('Filter Options')
selected_breeds = st.sidebar.multiselect('Select Breed(s)', options=df['Name'].unique())
selected_stat = st.sidebar.selectbox('Select Stat', options=['Friendly Rating (1-10)', 'Exercise Requirements (hrs/day)', 'Intelligence Rating (1-10)', 'Training Difficulty (1-10)'])
selected_exercise_requirement = st.sidebar.slider('Exercise Requirement (min)', 
                                                  min_value=int(df['Exercise Requirements (hrs/day)'].min()), 
                                                  max_value=int(df['Exercise Requirements (hrs/day)'].max()), 
                                                  value=(int(df['Exercise Requirements (hrs/day)'].min()), 
                                                         int(df['Exercise Requirements (hrs/day)'].max())))
selected_friendly_rating = st.sidebar.slider('Friendly Rating (min)', 
                                             min_value=int(df['Friendly Rating (1-10)'].min()), 
                                             max_value=int(df['Friendly Rating (1-10)'].max()), 
                                             value=(int(df['Friendly Rating (1-10)'].min()), 
                                                    int(df['Friendly Rating (1-10)'].max())))
selected_intelligence_rating = st.sidebar.slider('Intelligence Rating (1-10) (min)', 
                                                 min_value=int(df['Intelligence Rating (1-10)'].min()), 
                                                 max_value=int(df['Intelligence Rating (1-10)'].max()), 
                                                 value=(int(df['Intelligence Rating (1-10)'].min()), 
                                                        int(df['Intelligence Rating (1-10)'].max())))
selected_training_difficulty = st.sidebar.slider('Training Difficulty (min)', 
                                                 min_value=int(df['Training Difficulty (1-10)'].min()), 
                                                 max_value=int(df['Training Difficulty (1-10)'].max()), 
                                                 value=(int(df['Training Difficulty (1-10)'].min()), 
                                                        int(df['Training Difficulty (1-10)'].max())))

# Filter dataset based on selection
filtered_df = df[(df['Name'].isin(selected_breeds)) & 
                 (df['Exercise Requirements (hrs/day)'] >= selected_exercise_requirement[0]) & 
                 (df['Exercise Requirements (hrs/day)'] <= selected_exercise_requirement[1]) &
                 (df['Friendly Rating (1-10)'] >= selected_friendly_rating[0]) &
                 (df['Friendly Rating (1-10)'] <= selected_friendly_rating[1]) &
                 (df['Intelligence Rating (1-10)'] >= selected_intelligence_rating[0]) &
                 (df['Intelligence Rating (1-10)'] <= selected_intelligence_rating[1]) &
                 (df['Training Difficulty (1-10)'] >= selected_training_difficulty[0]) &
                 (df['Training Difficulty (1-10)'] <= selected_training_difficulty[1])]

# If no breeds are selected, display the entire dataset
if not selected_breeds:
    filtered_df = df

# Dashboard title
st.title('Dog Breeds Dashboard')

# Display filtered information
if selected_breeds:
    st.write(f"### Details for Selected Breed(s)")
    st.write(filtered_df)

# Plotly interactive graph - Exercise vs Training Difficulty with jitter added
jittered_x = filtered_df['Exercise Requirements (hrs/day)'] + np.random.uniform(-0.1, 0.1, size=len(filtered_df))
fig1 = px.scatter(
    filtered_df, x=jittered_x, y='Training Difficulty (1-10)', color='Name', 
    title='Exercise Requirements vs Training Difficulty', labels={"x": "Exercise Requirements (hrs/day) (with jitter)", "y": "Training Difficulty (1-10)"}
)
fig1.update_traces(marker=dict(size=10, opacity=0.7))
st.plotly_chart(fig1)

# Calculate Pearson Correlation for Exercise vs Training Difficulty
if len(filtered_df['Exercise Requirements (hrs/day)'].unique()) > 1 and len(filtered_df['Training Difficulty (1-10)'].unique()) > 1:
    pearson_corr, p_value = pearsonr(filtered_df['Exercise Requirements (hrs/day)'], filtered_df['Training Difficulty (1-10)'])
    slope, intercept = np.polyfit(filtered_df['Exercise Requirements (hrs/day)'], filtered_df['Training Difficulty (1-10)'], 1)
    st.write(f"""### Pearson Correlation for Exercise Requirements vs Training Difficulty: {pearson_corr:.2f} (p-value: {p_value:.4f})
Slope of Best Fit Line: {slope:.2f}""")

# Plotly interactive graph - Friendliness vs Selected Stat
if selected_stat in filtered_df.columns:
    fig2 = px.bar(filtered_df, x='Name', y=selected_stat, title=f'Friendliness vs {selected_stat}')
    st.plotly_chart(fig2)

# Calculate Pearson Correlation for Friendliness vs Intelligence
if len(filtered_df['Friendly Rating (1-10)'].unique()) > 1 and len(filtered_df['Intelligence Rating (1-10)'].unique()) > 1:
    pearson_corr_fi, p_value_fi = pearsonr(filtered_df['Friendly Rating (1-10)'], filtered_df['Intelligence Rating (1-10)'])
    slope_fi, intercept_fi = np.polyfit(filtered_df['Friendly Rating (1-10)'], filtered_df['Intelligence Rating (1-10)'], 1)
    st.write(f"""### Pearson Correlation for Friendliness vs Intelligence: {pearson_corr_fi:.2f} (p-value: {p_value_fi:.4f})
Slope of Best Fit Line: {slope_fi:.2f}""")

    # Additional visualization for Friendliness vs Intelligence
    fig3 = px.scatter(
        filtered_df, x='Friendly Rating (1-10)', y='Intelligence Rating (1-10)', color='Name', 
        title='Friendliness Rating vs Intelligence Rating (1-10)', labels={"x": "Friendly Rating (1-10)", "y": "Intelligence Rating (1-10)"}
    )
    fig3.update_traces(marker=dict(size=10, opacity=0.7))
    st.plotly_chart(fig3)

# Echarts example - Friendliness Rating Distribution
echarts_option = {
    "title": {
        "text": "Friendliness Rating Distribution",
    },
    "tooltip": {},
    "xAxis": {
        "type": 'category',
        "data": filtered_df['Name'].tolist(),
    },
    "yAxis": {
        "type": 'value'
    },
    "series": [
        {
            "data": filtered_df['Friendly Rating (1-10)'].tolist(),
            "type": 'bar'
        }
    ]
}
st_echarts(options=echarts_option)

# Summary statistics for the selected breeds
if selected_breeds:
    st.write(f"### Summary Statistics for Selected Breed(s)")
    st.write(filtered_df.describe())
