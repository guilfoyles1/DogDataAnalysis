import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_echarts import st_echarts
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/Users/shaynaguilfoyle/Final349/data/Dog Breads Around The World.csv')

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

# Correlation Matrix
st.write("### Correlation Matrix")
corr_matrix = df[['Friendly Rating (1-10)', 'Exercise Requirements (hrs/day)', 'Intelligence Rating (1-10)', 'Training Difficulty (1-10)']].corr()
fig_corr, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)

# Clustering with KMeans
st.write("### KMeans Clustering")
num_clusters = st.slider('Select Number of Clusters', min_value=2, max_value=10, value=3)
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(df[['Friendly Rating (1-10)', 'Exercise Requirements (hrs/day)', 'Intelligence Rating (1-10)', 'Training Difficulty (1-10)']])
df['Cluster'] = kmeans.labels_
fig_cluster = px.scatter_3d(df, x='Friendly Rating (1-10)', y='Exercise Requirements (hrs/day)', z='Intelligence Rating (1-10)', color='Cluster', title='KMeans Clustering of Dog Breeds')
st.plotly_chart(fig_cluster)

# Prediction Model - Linear Regression
st.write("### Prediction Model: Training Difficulty Prediction")
X = df[['Exercise Requirements (hrs/day)', 'Friendly Rating (1-10)', 'Intelligence Rating (1-10)']]
y = df['Training Difficulty (1-10)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

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

# Add regression analysis graphs at the end
st.write("### Regression Analysis Results")

# Exercise Requirements vs Training Difficulty
X_exercise = df[['Exercise Requirements (hrs/day)']]
y_training = df['Training Difficulty (1-10)']
X_train_exercise, X_test_exercise, y_train_training, y_test_training = train_test_split(X_exercise, y_training, test_size=0.2, random_state=42)
model_exercise = LinearRegression()
model_exercise.fit(X_train_exercise, y_train_training)
y_pred_exercise = model_exercise.predict(X_test_exercise)
fig_exercise, ax_exercise = plt.subplots()
ax_exercise.scatter(X_test_exercise, y_test_training, color='blue', label='Actual')
sorted_idx_exercise = np.argsort(X_test_exercise.values.flatten())
sorted_X_exercise = X_test_exercise.values.flatten()[sorted_idx_exercise]
sorted_y_pred_exercise = y_pred_exercise[sorted_idx_exercise]
ax_exercise.plot(sorted_X_exercise, sorted_y_pred_exercise, color='red', label='Regression Line')
ax_exercise.set_title("Exercise Requirements vs Training Difficulty")
ax_exercise.set_xlabel("Exercise Requirements (hrs/day)")
ax_exercise.set_ylabel("Training Difficulty (1-10)")
ax_exercise.legend()
st.pyplot(fig_exercise)

# Friendliness vs Intelligence
X_friend = df[['Friendly Rating (1-10)']]
y_intel = df['Intelligence Rating (1-10)']
X_train_friend, X_test_friend, y_train_intel, y_test_intel = train_test_split(X_friend, y_intel, test_size=0.2, random_state=42)
model_friend = LinearRegression()
model_friend.fit(X_train_friend, y_train_intel)
y_pred_friend = model_friend.predict(X_test_friend)
fig_friend, ax_friend = plt.subplots()
ax_friend.scatter(X_test_friend, y_test_intel, color='blue', label='Actual')
sorted_idx_friend = np.argsort(X_test_friend.values.flatten())
sorted_X_friend = X_test_friend.values.flatten()[sorted_idx_friend]
sorted_y_pred_friend = y_pred_friend[sorted_idx_friend]
ax_friend.plot(sorted_X_friend, sorted_y_pred_friend, color='red', label='Regression Line')
ax_friend.set_title("Friendliness vs Intelligence")
ax_friend.set_xlabel("Friendly Rating (1-10)")
ax_friend.set_ylabel("Intelligence Rating (1-10)")
ax_friend.legend()
st.pyplot(fig_friend)
