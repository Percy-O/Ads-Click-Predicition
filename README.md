
# 🧠 Ads Click Prediction Using Machine Learning

This project focuses on predicting whether a user will click on an online advertisement based on various behavioral and demographic features. The primary objective is to build a machine learning model that can accurately classify users into two categories: those who will click on an ad and those who will not.

---

## 📌 Project Overview

In the world of digital advertising, predicting user engagement is key to optimizing ad placements and improving marketing strategies. This project utilizes a dataset of user behavior and demographic information to train a machine learning model that predicts the likelihood of ad clicks.

The project includes the full data science pipeline:

* Data loading and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training and evaluation
* Model deployment using joblib
* Final prediction simulation

---

## 📂 Dataset

The dataset used in this project contains the following features:

* `Daily Time Spent on Site`
* `Age`
* `Area Income`
* `Daily Internet Usage`
* `Timestamp`
* `Country`
* `Ad Topic Line`
* `Male` (binary gender)
* `Clicked on Ad` (target variable)

📝 Source: Provided as a `.csv` file in `drive/MyDrive/ads_click/Ad_Click_Data.csv`.

---

## 🔧 Tools & Technologies Used

* **Python**
* **NumPy**, **Pandas** – data manipulation
* **Seaborn**, **Matplotlib**, **Plotly** – data visualization
* **Scikit-learn** – machine learning
* **Joblib** – model persistence
* **Google Colab** – development environment

---

## 📊 Exploratory Data Analysis (EDA)

The dataset was thoroughly explored through:

* Distribution analysis of the target variable
* Country-wise ad click insights
* Time-based user behavior (hour, weekday)
* Pairwise relationships between features
* Age and gender effects on ad clicks
* Correlation heatmaps for feature selection

---

## ⚙️ Feature Engineering

* Converted `Timestamp` to datetime format
* Extracted new time-based features: `Hour`, `Day`, `Month`, `Weekday`
* Removed null values and dropped non-contributing columns

---

## 🧪 Model Training & Evaluation

* **Model Used:** `RandomForestClassifier` from scikit-learn
* **Train/Test Split:** 70/30
* **Evaluation Metrics:**

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * Confusion Matrix visualization

---

## 🎯 Results

The Random Forest model achieved strong classification performance. Key evaluation results were visualized using seaborn and matplotlib, and the model was exported as `model.pkl` for future use.

---

## 🔮 Prediction Simulation

The saved model was reloaded using `joblib` and tested with simulated user data:

```python
data = pd.Series([
    [10, 2000, 50000, 1200],
    [60, 2400, 54440000, 200],
    [17, 200, 50000, 500]
])
```

The predictions classified each user as either likely to click the ad or not.

---

## 📁 Project Structure

```
ads_click_prediction/
│
├── ads_click_prediction.py      # Main project script
├── model.pkl                    # Saved ML model
├── README.md                    # Project documentation
└── Ad_Click_Data.csv            # Dataset (not included in repo, stored in Google Drive)
```

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/ads-click-prediction.git
   cd ads-click-prediction
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python script:

   ```bash
   python ads_click_prediction.py
   ```

---

## 📌 Future Improvements

* Add a web interface to accept user inputs for prediction
* Extend to more advanced models like XGBoost or Neural Networks
* Deploy the model via Flask or FastAPI
* Enable real-time ad click predictions

---

## 🙌 Acknowledgements

* Dataset preprocessing and feature extraction inspired by common practices in behavioral analysis.
* Visualization support via Seaborn, Matplotlib and Plotly.

---

## 📬 Contact

**Author:** Percy Owoeye
**GitHub:** [@yourusername](https://github.com/percy-o)
**Email:** [owoeyepercyolawale@gmail.com](mailto:owoeyepercyolawale@gmail.com)

Feel free to star ⭐ this repo if you found it helpful!

---

Let me know if you’d like this turned into a markdown file or customized further with deployment instructions or Jupyter notebook links.
