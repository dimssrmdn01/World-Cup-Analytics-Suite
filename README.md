#  World Cup Analytics Suite

An integrated, end-to-end data science ecosystem for sports analytics. This suite combines Supervised Learning, Unsupervised Learning, and Natural Language Processing (NLP) into three interactive Streamlit web applications.

##  Modules Overview

### 1.  Match Outcome Predictor (`app.py`)
A predictive engine that calculates the probability of match outcomes (Home Win, Draw, Away Win) based on team power ratings and home advantage.
* **Algorithm:** Logistic Regression with L1 (Lasso) Penalty.
* **Features:** Dynamic probability meters, custom UI CSS, and interactive team selection.

### 2.  Advanced Scout Terminal (`tracker.py`)
An unsupervised learning dashboard designed to discover hidden playstyle patterns among football players based on their performance metrics (Goals, Assists, Pass Accuracy, Tackles, Stamina).
* **Algorithm:** K-Means Clustering & Principal Component Analysis (PCA).
* **Features:** Cyber-scout themed UI, 2D interactive galaxy mapping (Plotly), and automated role grouping.

### 3.  Live Sentiment Radar (`sentiment.py`)
A real-time public opinion tracker that simulates and analyzes social media feeds during a match to gauge fan sentiment.
* **Engine:** TextBlob (Natural Language Processing).
* **Features:** Live timeline charts, polarity scoring (Positive/Negative/Neutral), and a real-time simulated stream feed.

##  Tech Stack
* **Language:** Python 3
* **Frontend:** Streamlit, HTML/CSS (Custom Styling)
* **Machine Learning:** Scikit-Learn (Logistic Regression, K-Means, PCA, StandardScaler)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Plotly Express
* **NLP:** TextBlob

##  Installation & Usage

1. Clone this repository:
```bash
git clone https://github.com/dimssrmdn01/sports-analytics-suite.git
cd sports-analytics-suite
```

2. Install the required dependencies:
```bash
pip install streamlit pandas numpy scikit-learn plotly textblob
```

3. Run the applications (open separate terminals for each, or run one at a time):
```bash
streamlit run app.py         # Launch Match Predictor
streamlit run tracker.py     # Launch Scout Terminal
streamlit run sentiment.py   # Launch Sentiment Radar
```

## 👨‍💻 Author
**Dimas Arya Ramadhan**
Data Science Enthusiast | Building intelligent systems through data.
