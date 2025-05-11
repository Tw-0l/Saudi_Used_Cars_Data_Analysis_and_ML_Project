# Saudi Used Cars: Market Analysis & Price Prediction

![Saudi Cars](https://img.shields.io/badge/Saudi-Used%20Cars-darkgreen?style=for-the-badge)
![Data Science](https://img.shields.io/badge/Data%20Science-Project-blue?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Price%20Prediction-orange?style=for-the-badge)

## 🚗 Project Overview
A comprehensive data science project that analyzes the Saudi Arabian used car market and builds machine learning models to predict vehicle prices. This project offers insights into market trends, buyer preferences, and the factors that most significantly influence used car valuations in Saudi Arabia.

## 🔍 Business Value
- Help **car sellers** set optimal pricing strategies
- Assist **car buyers** in identifying fair market values
- Provide **dealerships** with market intelligence
- Enable **market analysts** to understand trends in the Saudi automotive sector

## 📊 Data Analysis Highlights

### Market Insights
- Distribution of car brands, models, and manufacturing years
- Regional price variations across Saudi Arabia
- Most popular vehicle categories and configurations
- Seasonal pricing trends and market fluctuations

### Price Determinants
- Correlation between vehicle age and price depreciation
- Impact of mileage on resale value
- Premium paid for specific brands and luxury features
- Influence of fuel type and transmission type on pricing

## 🧠 Machine Learning Models

### Prediction Tasks
- Used car price prediction based on vehicle attributes
- Market segment classification
- Time-to-sell estimation
- Anomaly detection for identifying undervalued/overvalued listings

### Implemented Models
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting (XGBoost)
- Support Vector Regression
- Deep Learning approaches (for complex feature interactions)

### Performance Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Model-specific explainability metrics (SHAP values)

## 📈 Key Findings
- Toyota and Hyundai dominate the Saudi used car market with the highest listing volumes
- Cars lose approximately X% of their value in the first year, with depreciation slowing to Y% annually after 5 years
- SUVs retain value better than sedans in the Saudi market
- Regional pricing differences show cars in Riyadh and Jeddah command premium prices
- Mileage impact varies significantly by brand, with luxury brands showing higher sensitivity
- Vehicles with service history documentation command a X% premium on average

## 🛠️ Technologies Used
- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning implementation
- **XGBoost/LightGBM**: Gradient boosting frameworks
- **SHAP**: Model interpretation
- **Flask/Streamlit**: Web application for model deployment (optional)
- **Jupyter Notebooks**: Interactive development environment

## 📁 Repository Structure
```
Saudi_Used_Cars_Data_Analysis_and_ML_Project/
├── data/
│   ├── raw/                      # Original scraped/collected dataset
│   ├── processed/                # Cleaned and transformed data
│   └── external/                 # Additional data sources (e.g., economic indicators)
├── notebooks/
│   ├── 01_data_exploration.ipynb # Initial EDA
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── data/                     # Data processing scripts
│   ├── features/                 # Feature engineering
│   ├── models/                   # Model implementation
│   └── visualization/            # Visualization utilities
├── models/                       # Saved model files
├── reports/                      # Analysis reports and figures
│   ├── figures/                  # Generated graphics
│   └── insights/                 # Key findings documents
├── app/                          # Web application (if implemented)
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone the repository
   ```bash
   git clone https://github.com/Tw-0l/Saudi_Used_Cars_Data_Analysis_and_ML_Project.git
   cd Saudi_Used_Cars_Data_Analysis_and_ML_Project
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install required dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run Jupyter to explore notebooks
   ```bash
   jupyter notebook
   ```

## 💻 Usage Examples

### Price Prediction
```python
# Example code for predicting car prices
from src.models.predictor import CarPricePredictor

# Initialize the predictor with the trained model
predictor = CarPricePredictor(model_path='models/best_model.pkl')

# Car details
car_details = {
    'brand': 'Toyota',
    'model': 'Camry',
    'year': 2018,
    'kilometers': 75000,
    'transmission': 'Automatic',
    'region': 'Riyadh',
    # Additional features...
}

# Get prediction
predicted_price = predictor.predict(car_details)
print(f"Predicted price: SAR {predicted_price:,.2f}")
```

### Market Analysis
```python
# Example code for analyzing market trends
from src.visualization.market_analyzer import MarketTrendAnalyzer

analyzer = MarketTrendAnalyzer('data/processed/clean_data.csv')

# Generate brand market share visualization
analyzer.plot_brand_market_share()

# Analyze price trends by car age
analyzer.plot_price_by_age('Toyota', 'Camry')
```

## 📊 Interactive Dashboard (Optional)
The project includes an interactive web dashboard built with Streamlit that allows users to:
- Input vehicle details and get instant price predictions
- Explore market trends with interactive visualizations
- Compare prices across different regions and car models
- View historical price trends for specific vehicle types

To run the dashboard:
```bash
cd app
streamlit run app.py
```

## 🌐 Saudi Market Specifics
This project accounts for unique aspects of the Saudi used car market:

- **Import Regulations**: Impact of Saudi vehicle import policies on pricing
- **Desert Environment**: Consideration of how harsh climate affects vehicle conditions
- **Fuel Economy**: Lower influence of fuel economy due to subsidized fuel prices
- **Regional Preferences**: Strong preferences for specific brands and vehicle types
- **Religious Calendar**: Market activity fluctuations related to Ramadan and Hajj seasons

## 🔮 Future Enhancements
- Integration with real-time market data from major Saudi automotive marketplaces
- Addition of image-based valuation using computer vision techniques
- Time series forecasting for market trend prediction
- Expansion to include commercial vehicles and luxury market segments
- Development of a mobile application for on-the-go valuations

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to suggest improvements or additional analyses.

## 📊 Data Sources
- Major Saudi automotive marketplaces
- Public vehicle registration data
- Market research reports on the Saudi automotive sector
- *Note: Specific data sources and collection methodologies are detailed in the data documentation*
  

## 📞 Contact
- GitHub: [@Tw-0l](https://github.com/Tw-0l)

---

*This project aims to bring transparency and data-driven insights to the Saudi used car market, helping consumers and businesses make informed decisions based on comprehensive market analysis and accurate price predictions.*
