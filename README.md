# PropIQ · House Price Intelligence

> My first Machine Learning project — built to understand how ML actually works, one step at a time.

---

## About This Project

I am currently a student exploring the world of Data Science and Machine Learning for the very first time. Instead of jumping straight into complex problems, I decided to start small and really understand the fundamentals before scaling up.

This project is a **Linear Regression model** that predicts property prices (in Lakhs ₹) based on area size (sq. ft.). I intentionally used a small 10-row dataset because I wanted to focus on learning the complete ML workflow — data preparation, model training, evaluation, and deployment — rather than just getting results.

The model is wrapped in an interactive Streamlit dashboard called **PropIQ**, where you can enter any area size and instantly get a predicted property price.

---

## What I Learned

This was my first time working with Machine Learning and I came away understanding:

- How to structure and prepare data for an ML model
- What feature scaling is and why StandardScaler matters
- How Linear Regression finds the relationship between two variables
- What R², MAE, and RMSE actually mean and how to interpret them
- Why cross-validation is important even on small datasets
- How to build and deploy an interactive ML app using Streamlit

---

## Features

- 🏡 Interactive price estimator — enter an area, get a predicted price
- 📈 Regression plot — visualises the relationship between area and price
- 📊 Model metrics — R², MAE, RMSE and Cross-Validated R²
- 📐 Area percentile indicator — shows where your input sits in the dataset
- 📂 Dataset explorer — view the raw data and descriptive statistics
- 🔬 Actual vs Predicted table — compares model output against real values

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.x |
| Machine Learning | scikit-learn |
| Data Handling | pandas, numpy |
| Visualisation | matplotlib |
| App Framework | Streamlit |

---

## Project Structure

```
PropIQ-House-Price-Intelligence/
│
├── app.py                              # Streamlit application
├── Linear_Regression_Area_Price.ipynb  # Model development notebook
├── requirements.txt                    # Dependencies
│
├── data/
│   └── area_price_dataset.csv          # 10-sample training dataset
│
└── assets/
    └── PropIQ Screenshot.png                  # App preview
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/juhi-m1711/PropIQ-House-Price-Intelligence.git
cd PropIQ-House-Price-Intelligence
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## Model Details

| Metric | Value |
|---|---|
| Algorithm | Linear Regression |
| Feature Scaling | StandardScaler |
| Evaluation | Train/Test Split + 5-Fold Cross-Validation |
| Input Feature | Area (sq. ft.) |
| Target Variable | Price (Lakhs ₹) |

---

## Why Only 10 Rows of Data?

When I started this project, I came across a lot of advice saying "get a bigger dataset." But I think starting small was the right call for a first project.

With just 10 rows, I could see exactly what was happening at every step. I could trace how the data flowed through the pipeline, understand what each metric was reacting to, and actually debug things when they went wrong. A large dataset would have given me results, but not necessarily understanding.

As I grow more confident, I plan to rebuild this with real-world data and more features — but the foundation I built here will make that a lot easier.

---

## What's Next

- [ ] Add more features — bedrooms, location, floor, age of property
- [ ] Try Polynomial Regression and compare results
- [ ] Source a larger real-world housing dataset
- [ ] Deploy live on Streamlit Community Cloud

---

## A Note to Fellow Beginners

If you are also just starting out with ML — don't wait for the perfect project or the perfect dataset. Build something small, understand it fully, and then build something bigger. That's exactly what this project was for me.

---

## Author

**Juhi** · Student & Aspiring Data Scientist  
[GitHub](https://github.com/juhi-m1711) · [LinkedIn](https://www.linkedin.com/in/juhi-moudekar/)

---

*Project #1 of many. Built to learn, shared to inspire.*
