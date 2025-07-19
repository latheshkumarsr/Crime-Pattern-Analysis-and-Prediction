# Crime Prediction using Machine Learning

## Project Overview

This project focuses on developing a machine learning-based system to analyze historical crime data, identify patterns, and predict future crime incidents. The goal is to provide law enforcement agencies with actionable insights to enable proactive policing, optimize resource allocation, and enhance public safety.

Traditional crime analysis methods are often reactive and struggle with large, complex datasets. This project leverages modern machine learning algorithms to offer a data-driven, scalable, and objective approach to uncover hidden patterns and trends in crime data, ultimately contributing to preventive policing and strategic planning.

## Features

*   **Historical Data Analysis:** Analyzes past crime data to extract meaningful patterns and trends.
*   **Machine Learning Models:** Develops and utilizes various machine learning models for crime prediction.
*   **Hotspot Identification:** Identifies crime-prone areas (hotspots) using clustering techniques.
*   **Crime Type Classification:** Classifies different types of crimes based on historical data.
*   **Performance Evaluation:** Evaluates and compares the performance of different algorithms using standard metrics.
*   **Data-Driven Insights:** Provides insights to assist law enforcement in decision-making and resource management.
*   **Modular Architecture:** Designed with distinct modules for data acquisition, preprocessing, model training, prediction, and visualization.
*   **User-Friendly Visualization:** Presents prediction results and crime patterns through intuitive charts, graphs, and maps.

## Technologies Used

*   **Programming Language:** Python 3.x
*   **Core Libraries:**
    *   `pandas`: For data manipulation and preprocessing.
    *   `numpy`: For numerical operations.
    *   `scikit-learn`: For machine learning model building (Decision Tree, Random Forest, SVM, K-Means, KNN) and evaluation.
*   **Visualization Libraries:**
    *   `matplotlib`: For basic plotting and customization.
    *   `seaborn`: For creating attractive and informative statistical graphics (heatmaps, scatter plots).
*   **Development Environment:**
    *   `Jupyter Notebook`: For interactive development, experimentation, and documentation.
    *   `VS Code` / `PyCharm` (optional): For integrated development.
*   **Dataset:** Publicly available crime datasets (e.g., from Kaggle, Chicago Crime Dataset).
*   **Version Control:** Git (recommended for collaborative development).

## System Architecture

The system follows a modular architecture to ensure scalability, maintainability, and usability:

1.  **Data Acquisition Module:** Collects historical crime data from public sources.
2.  **Preprocessing Module:** Cleans and transforms raw data (handling missing values, encoding categorical variables, normalizing features).
3.  **Model Training Module:** Trains machine learning models (Decision Tree, Random Forest, SVM) using the preprocessed data.
4.  **Prediction Module:** Uses trained models to predict crime categories or hotspots based on input parameters.
5.  **Visualization Module:** Presents prediction results and crime patterns using various visual aids.
6.  **User Interface:** Allows users to input parameters and view outputs in a user-friendly format.

![System Architecture Diagram](https://github.com/latheshkumarsr/Crime-Pattern-Analysis-and-Prediction/blob/main/System%20Architecture%20Diagram.png)


## Machine Learning Models Implemented

The project explores and implements several machine learning algorithms:

*   **Decision Tree Classifier:** Interpretable, handles mixed data types, but prone to overfitting.
*   **Random Forest Classifier:** Ensemble method offering high accuracy, robustness to noise, and good handling of missing values. (Identified as the best performer in evaluations).
*   **Support Vector Machine (SVM):** Effective for high-dimensional and non-linearly separable data, but can be computationally intensive for large datasets.
*   **K-Means Clustering:** An unsupervised method used as a supplementary layer to identify crime hotspots and visualize spatial patterns.
*   **K-Nearest Neighbors (KNN):** A simple, instance-based learning algorithm used for classification.

## Setup and Installation

To set up the project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/crime-prediction.git
cd crime-prediction
```
### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
- Note: If requirements.txt is not provided, you can create one by listing the libraries mentioned in the Technologies Used section:
```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```
### 4. Obtain Dataset
Download a publicly available crime dataset (e.g., from Kaggle or a government crime database). 
Ensure the dataset is in CSV format and place it in the project's root directory or a designated `data/` folder.

Example: `crime_dataset.csv`

### Usage
## 1. Open Jupyter Notebook
```bash
jupyter notebook
```
## 2. Navigate and Run Notebooks
Open the relevant Jupyter Notebook files (e.g., `crime_prediction_analysis.ipynb`,` model_training.ipynb`) to explore the data, run the preprocessing steps, train models, and visualize results.

### Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## output
![Represtion by Confusion Matrix](https://github.com/latheshkumarsr/Crime-Pattern-Analysis-and-Prediction/blob/main/7.png)
![Represtion by Map_ploting](https://github.com/latheshkumarsr/Crime-Pattern-Analysis-and-Prediction/blob/main/9.png)
