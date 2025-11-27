# 🍷 Wine Classification AI

An elegant, AI-powered web application that predicts wine cultivars using machine learning analysis of chemical properties.

![Wine Classification](https://img.shields.io/badge/Accuracy-98%25-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)
![License](https://img.shields.io/badge/License-Academic-yellow)

## 🎯 What It Does

Upload wine chemical analysis data or use interactive sliders → Get instant AI prediction of wine type with confidence scores and beautiful visualizations!

## ✨ Features

- 🎨 **Beautiful Modern UI** - Gradient designs, interactive charts, professional styling
- 🎚️ **Interactive Sliders** - Adjust 13 wine features with real-time feedback
- 📊 **Visual Analytics** - Plotly charts showing probability distributions and chemical profiles
- 📁 **File Upload** - Support for JSON input files
- 🤖 **Ensemble AI Model** - Combines SVM, Gradient Boosting, and Random Forest
- 📈 **98%+ Accuracy** - Highly reliable predictions
- 🎭 **Dual Input Modes** - Choose between manual sliders or file upload

## 🚀 Quick Start

### Installation

1. **Install Required Packages**
```bash
pip install streamlit scikit-learn numpy plotly
```

2. **Train the Model**
```bash
cd Streamlit_Labs/src
python train_wine.py
```

3. **Launch the App**
```bash
streamlit run Dashboard.py
```

4. **Open Your Browser**
   - App opens automatically at `http://localhost:8501`

## 💻 How to Use

### Method 1: Interactive Sliders

1. Select "🎚️ Interactive Sliders" in the sidebar
2. Adjust the 13 wine feature sliders
3. Click "🔮 Predict Wine Class"
4. View your results with confidence scores and visualizations!

### Method 2: Upload JSON File

1. Select "📄 Upload JSON File" in the sidebar
2. Upload your JSON file (format below)
3. Click "🔮 Predict Wine Class"
4. See instant results!

### JSON Input Format
```json
{
  "features": [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]
}
```

**Must include exactly 13 features in this order:**
1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315
13. Proline

## 📁 Project Structure
```
Streamlit_Labs/
├── src/
│   ├── Dashboard.py          # Main app with beautiful UI
│   ├── train_wine.py          # Ensemble model training
│   ├── wine_model.pkl         # Trained AI model (generated)
│   └── __init__.py
├── data/
│   └── wine_test.json         # Sample test files
├── assets/                    # UI assets
└── README.md                  # This file
```

## 🧪 Sample Test Data

### Sample 1: Class 0 Wine (High Confidence)
```json
{
  "features": [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]
}
```
Expected: Class 0 (Cultivar 1) - Rich in alcohol and proline

### Sample 2: Class 1 Wine
```json
{
  "features": [12.37, 1.63, 2.3, 24.5, 88.0, 2.22, 2.45, 0.4, 1.9, 2.12, 0.89, 2.78, 342.0]
}
```
Expected: Class 1 (Cultivar 2) - Balanced profile

### Sample 3: Class 2 Wine
```json
{
  "features": [13.49, 3.59, 2.19, 19.5, 88.0, 1.62, 0.48, 0.58, 0.88, 5.7, 0.81, 1.82, 580.0]
}
```
Expected: Class 2 (Cultivar 3) - Higher malic acid

## 🎨 Visual Features

### What You'll See:

- **📊 Probability Bar Chart** - Shows confidence for each wine class
- **🎯 Radar Chart** - Visual representation of wine's chemical profile
- **💳 Prediction Cards** - Large, colorful results with wine characteristics
- **📈 Confidence Metrics** - Real-time probability percentages
- **🎨 Gradient Design** - Professional wine-themed color scheme

## 🤖 AI Model Details

### Ensemble Architecture

The app uses a **Voting Classifier** combining three powerful algorithms:

| Algorithm | Test Accuracy | CV Accuracy | Specialty |
|-----------|--------------|-------------|-----------|
| SVM (RBF kernel) | 100% | 98.89% | Pattern recognition |
| Gradient Boosting | 97.22% | 96.67% | Feature interactions |
| Random Forest | 97.22% | 97.78% | Robust predictions |
| **Ensemble** | **100%** | **98.33%** | **Best overall** |

### Why Ensemble?

- Combines strengths of multiple models
- Reduces individual model weaknesses
- More reliable predictions
- Better generalization

## 📊 Wine Classes

| Class | Name | Characteristics |
|-------|------|----------------|
| 🍷 Class 0 | Cultivar 1 | High alcohol, high proline, bold flavanoids |
| 🍇 Class 1 | Cultivar 2 | Balanced profile, moderate in all features |
| 🍾 Class 2 | Cultivar 3 | Higher malic acid, lower flavanoids |

## 🔧 Troubleshooting

### ❌ "Model not found"
**Solution:** Run `python train_wine.py` from the `src` folder

### ❌ "ModuleNotFoundError: No module named 'plotly'"
**Solution:** `pip install plotly`

### ❌ "Expected 13 features, got X"
**Solution:** Ensure your JSON has exactly 13 numbers in correct order

### ❌ "st.set_page_config() error"
**Solution:** Already fixed in current version - make sure you have latest code

### ❌ Can't access the app
**Solution:** Check terminal for URL (usually `http://localhost:8501`)

### ❌ Port already in use
**Solution:** `streamlit run Dashboard.py --server.port 8502`

## 🎓 Technical Stack

- **Frontend:** Streamlit
- **Visualizations:** Plotly
- **ML Framework:** scikit-learn
- **Data Processing:** NumPy, Pandas
- **Dataset:** UCI Wine Recognition Dataset (178 samples, 13 features)

## 📈 Performance Metrics

- **Training Accuracy:** 100%
- **Cross-Validation:** 98.33% (±1.67%)
- **Prediction Time:** < 0.1 seconds
- **Model Size:** ~50KB
- **Features:** 13 chemical properties
- **Classes:** 3 wine cultivars

## 🌟 Key Highlights

✅ Professional, modern UI with gradient designs  
✅ Dual input methods (sliders + file upload)  
✅ Real-time visual analytics  
✅ Interactive radar and bar charts  
✅ Color-coded wine class information  
✅ 98%+ prediction accuracy  
✅ Fast, responsive interface  
✅ Mobile-friendly design  

## 🔮 Future Enhancements

- [ ] Export prediction reports to PDF
- [ ] Batch prediction for multiple wines
- [ ] Historical prediction tracking
- [ ] SHAP value explanations
- [ ] Custom model retraining interface
- [ ] Confusion matrix visualization
- [ ] Feature importance analysis
- [ ] Support for custom datasets

## 📚 Learn More

### Understanding Wine Features

- **Alcohol:** Percentage of alcohol content (11.0-14.8%)
- **Malic Acid:** Tartness level (0.74-5.80 g/L)
- **Flavanoids:** Antioxidant compounds (0.34-5.08 mg/L)
- **Proline:** Amino acid content (278-1680 mg/L)
- **Color Intensity:** Visual richness (1.28-13.0)

### Dataset Source

This app uses the **UCI Wine Recognition Dataset**, a classic machine learning benchmark containing chemical analysis of wines from three different cultivars in Italy.

---