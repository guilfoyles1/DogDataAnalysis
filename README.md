# Dog Breeds Dashboard - README

## Overview

This project is a **Dog Breeds Dashboard** created using **Streamlit**, **Plotly**, and **pandas**. The dashboard allows users to explore, filter, and visualize a dataset of dog breeds, with a focus on characteristics such as exercise requirements, friendliness, intelligence, and training difficulty. Users can interactively select different breeds and view detailed visualizations, gaining insights about each breed's attributes.

## Features

- **Interactive Sidebar Filters**: Filter dog breeds based on parameters like Friendly Rating, Exercise Requirements, Intelligence Rating, and Training Difficulty.
- **Dynamic Visualizations**: Plotly and Echarts visualizations to explore relationships between various attributes.
- **Correlation Analysis**: Calculate and display Pearson correlation coefficients to understand the relationships between exercise, training difficulty, friendliness, and intelligence ratings.

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone tps://github.com/guilfoyles1/DogDataAnalysis
   ```

2. Navigate to the project directory:

   ```bash
   cd src
   ```

3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

   Creating a virtual environment helps to keep the dependencies required by the project isolated from your global Python environment, ensuring no conflicts with other projects.

4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   This command will install all necessary dependencies, such as **Streamlit**, **Plotly**, and others needed for the project.

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run src/dogs.py
   ```
2. Open your web browser and navigate to the given local URL (typically `http://localhost:8501`).

## Dataset

The dataset used for this dashboard is **"Dog Breeds Around The World"**, which provides information about various breeds, including:

- **Name**: The breed name
- **Exercise Requirements**: Average hours per day required for exercise
- **Friendly Rating**: Friendliness score on a scale of 1-10
- **Intelligence Rating**: Intelligence score on a scale of 1-10
- **Training Difficulty**: Difficulty rating of training, on a scale of 1-10

## Interactive Features

- **Breed Selection**: Choose from a list of breeds to view specific details.
- **Exercise Requirement Filter**: Use a slider to filter breeds by minimum and maximum daily exercise hours.
- **Friendliness & Intelligence Filters**: Adjust sliders to filter breeds based on friendliness and intelligence ratings.
- **Visualizations**:
  - **Scatter Plot**: Visualize Exercise Requirements vs. Training Difficulty with jitter added for clarity.
  - **Bar Chart**: Compare selected breeds based on various ratings.
  - **Correlation Calculations**: Display correlation statistics for understanding relationships between attributes.
  - **Echarts**: Visualize friendliness ratings in an interactive bar chart.

## File Structure

- **src/dogs.py**: Main script for the dashboard application.
- **data/**: Directory to hold the dataset (e.g., `Dog Breeds Around The World.csv`).
- **requirements.txt**: List of dependencies to run the application.

## Requirements

- **Python 3.7+**
- **Streamlit**
- **Plotly**
- **pandas**
- **numpy**
- **statsmodels**
- **scipy**

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## Notes

- The dataset file (`Dog Breeds Around The World.csv`) should be placed in the appropriate directory (`data/`), or you can modify the path in the script to match your setup.
- Ensure that you have the correct version of the libraries as per the requirements.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Contact

If you have questions or suggestions, please reach out to Shayna at [guilfoyles1@udayton.edu](mailto\:guilfoyles1@udayton.edu).

