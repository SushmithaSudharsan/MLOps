import json
import pickle
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger
from sklearn.datasets import load_wine
import plotly.graph_objects as go

# Model location - now in the same src folder
WINE_MODEL_LOCATION = Path(__file__).resolve().parent / 'wine_model.pkl'

# streamlit logger
LOGGER = get_logger(__name__)

# Load wine dataset to get min/max values
wine_data = load_wine()
feature_names = wine_data.feature_names
X = wine_data.data

# Calculate min and max for each feature
feature_ranges = {
    feature_names[i]: (float(X[:, i].min()), float(X[:, i].max()))
    for i in range(len(feature_names))
}

# Wine class descriptions
WINE_DESCRIPTIONS = {
    0: {
        "name": "Cultivar 1",
        "emoji": "🍷",
        "color": "#8B0000",
        "characteristics": "Rich in alcohol and proline, bold flavanoids"
    },
    1: {
        "name": "Cultivar 2",
        "emoji": "🍇",
        "color": "#9B59B6",
        "characteristics": "Balanced profile, moderate in all features"
    },
    2: {
        "name": "Cultivar 3",
        "emoji": "🍾",
        "color": "#E74C3C",
        "characteristics": "Higher malic acid, lower flavanoids"
    }
}


def load_model():
    """Load the wine model"""
    try:
        with open(WINE_MODEL_LOCATION, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None


def create_probability_chart(probabilities):
    """Create an attractive bar chart for probabilities"""
    colors = ['#8B0000', '#9B59B6', '#E74C3C']

    fig = go.Figure(data=[
        go.Bar(
            x=['Class 0', 'Class 1', 'Class 2'],
            y=[prob * 100 for prob in probabilities],
            marker_color=colors,
            text=[f'{prob*100:.1f}%' for prob in probabilities],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>'
        )
    ])

    fig.update_layout(
        title="Prediction Confidence Distribution",
        yaxis_title="Probability (%)",
        xaxis_title="Wine Class",
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        yaxis=dict(range=[0, 100], gridcolor='lightgray'),
    )

    return fig


def create_feature_radar(features, feature_names):
    """Create a radar chart showing feature values"""
    # Normalize features to 0-100 scale for visualization
    normalized_features = []
    for i, feature in enumerate(features):
        min_val, max_val = feature_ranges[feature_names[i]]
        normalized = ((feature - min_val) / (max_val - min_val)) * 100
        normalized_features.append(normalized)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=normalized_features,
        theta=[name.replace('_', ' ').title() for name in feature_names],
        fill='toself',
        fillcolor='rgba(139, 0, 0, 0.2)',
        line=dict(color='#8B0000', width=2),
        name='Wine Profile'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="Wine Chemical Profile",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def run():

    st.set_page_config(
        page_title="Wine Classification AI",
        page_icon="🍷",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            font-size: 3.5rem;
            font-weight: 700;
            text-align: center;
            background: linear-gradient(120deg, #8B0000, #E74C3C);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            padding: 1rem;
        }
        
        /* Subtitle styling */
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        
        /* Card styling */
        .prediction-card {
            padding: 2rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        /* Info box styling */
        .info-box {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f8f9fa;
            border-left: 5px solid #8B0000;
            margin: 1rem 0;
        }
        
        /* Metric styling */
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(120deg, #8B0000, #E74C3C);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(139,0,0,0.4);
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        }
        
        /* Feature slider labels */
        .stSlider label {
            font-weight: 600;
            color: #333;
        }
        
        /* Radio button styling */
        .stRadio > label {
            font-weight: 600;
            font-size: 1.1rem;
            color: #8B0000;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown('<h1 class="main-title">🍷 Wine Classification AI</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict wine cultivars using advanced machine learning</p>',
                unsafe_allow_html=True)

    # Add a divider
    st.markdown("---")

    # Build the sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")

        # Check model status with better visuals
        if WINE_MODEL_LOCATION.is_file():
            st.success("✅ AI Model Ready", icon="🤖")
        else:
            st.error("❌ Model Not Found", icon="⚠️")
            st.warning("Run `train_wine.py` first!")
            st.stop()

        st.markdown("---")

        # Input method selection with icons
        input_method = st.radio(
            "📥 Input Method",
            ["🎚️ Interactive Sliders", "📄 Upload JSON File"],
            help="Choose how to provide wine features"
        )

        features = []

        if input_method == "🎚️ Interactive Sliders":
            st.markdown("### 🔬 Wine Features")
            st.caption("Adjust sliders to set wine properties")

            with st.expander("⚗️ Chemical Composition", expanded=True):
                alcohol = st.slider(
                    "🍾 Alcohol",
                    min_value=feature_ranges['alcohol'][0],
                    max_value=feature_ranges['alcohol'][1],
                    value=(feature_ranges['alcohol'][0] +
                           feature_ranges['alcohol'][1]) / 2,
                    step=0.01
                )

                malic_acid = st.slider(
                    "🧪 Malic Acid",
                    min_value=feature_ranges['malic_acid'][0],
                    max_value=feature_ranges['malic_acid'][1],
                    value=(feature_ranges['malic_acid'][0] +
                           feature_ranges['malic_acid'][1]) / 2,
                    step=0.01
                )

                ash = st.slider(
                    "⚪ Ash",
                    min_value=feature_ranges['ash'][0],
                    max_value=feature_ranges['ash'][1],
                    value=(feature_ranges['ash'][0] +
                           feature_ranges['ash'][1]) / 2,
                    step=0.01
                )

                alcalinity_of_ash = st.slider(
                    "⚖️ Alcalinity of Ash",
                    min_value=feature_ranges['alcalinity_of_ash'][0],
                    max_value=feature_ranges['alcalinity_of_ash'][1],
                    value=(feature_ranges['alcalinity_of_ash'][0] +
                           feature_ranges['alcalinity_of_ash'][1]) / 2,
                    step=0.1
                )

                magnesium = st.slider(
                    "🧲 Magnesium",
                    min_value=feature_ranges['magnesium'][0],
                    max_value=feature_ranges['magnesium'][1],
                    value=(feature_ranges['magnesium'][0] +
                           feature_ranges['magnesium'][1]) / 2,
                    step=1.0
                )

            with st.expander("🌿 Phenolic Compounds", expanded=True):
                total_phenols = st.slider(
                    "💚 Total Phenols",
                    min_value=feature_ranges['total_phenols'][0],
                    max_value=feature_ranges['total_phenols'][1],
                    value=(feature_ranges['total_phenols'][0] +
                           feature_ranges['total_phenols'][1]) / 2,
                    step=0.01
                )

                flavanoids = st.slider(
                    "🌸 Flavanoids",
                    min_value=feature_ranges['flavanoids'][0],
                    max_value=feature_ranges['flavanoids'][1],
                    value=(feature_ranges['flavanoids'][0] +
                           feature_ranges['flavanoids'][1]) / 2,
                    step=0.01
                )

                nonflavanoid_phenols = st.slider(
                    "🍃 Nonflavanoid Phenols",
                    min_value=feature_ranges['nonflavanoid_phenols'][0],
                    max_value=feature_ranges['nonflavanoid_phenols'][1],
                    value=(feature_ranges['nonflavanoid_phenols'][0] +
                           feature_ranges['nonflavanoid_phenols'][1]) / 2,
                    step=0.01
                )

                proanthocyanins = st.slider(
                    "🔮 Proanthocyanins",
                    min_value=feature_ranges['proanthocyanins'][0],
                    max_value=feature_ranges['proanthocyanins'][1],
                    value=(feature_ranges['proanthocyanins'][0] +
                           feature_ranges['proanthocyanins'][1]) / 2,
                    step=0.01
                )

            with st.expander("🎨 Color Properties", expanded=True):
                color_intensity = st.slider(
                    "🌈 Color Intensity",
                    min_value=feature_ranges['color_intensity'][0],
                    max_value=feature_ranges['color_intensity'][1],
                    value=(feature_ranges['color_intensity'][0] +
                           feature_ranges['color_intensity'][1]) / 2,
                    step=0.01
                )

                hue = st.slider(
                    "🎭 Hue",
                    min_value=feature_ranges['hue'][0],
                    max_value=feature_ranges['hue'][1],
                    value=(feature_ranges['hue'][0] +
                           feature_ranges['hue'][1]) / 2,
                    step=0.01
                )

            with st.expander("🔬 Advanced Metrics", expanded=True):
                od280_od315 = st.slider(
                    "📊 OD280/OD315",
                    min_value=feature_ranges['od280/od315_of_diluted_wines'][0],
                    max_value=feature_ranges['od280/od315_of_diluted_wines'][1],
                    value=(feature_ranges['od280/od315_of_diluted_wines'][0] +
                           feature_ranges['od280/od315_of_diluted_wines'][1]) / 2,
                    step=0.01
                )

                proline = st.slider(
                    "💎 Proline",
                    min_value=feature_ranges['proline'][0],
                    max_value=feature_ranges['proline'][1],
                    value=(feature_ranges['proline'][0] +
                           feature_ranges['proline'][1]) / 2,
                    step=1.0
                )

            features = [
                alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
                total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
                color_intensity, hue, od280_od315, proline
            ]

            st.session_state["IS_INPUT_AVAILABLE"] = True
            st.session_state["features"] = features

        else:  # Upload JSON File
            st.markdown("### 📤 File Upload")
            test_input_file = st.file_uploader(
                'Choose a JSON file',
                type=['json'],
                help="Upload a JSON file with wine features"
            )

            if test_input_file:
                st.success("File uploaded successfully!")
                with st.expander("👁️ Preview Data"):
                    test_input_data = json.load(test_input_file)
                    st.json(test_input_data)

                if 'features' in test_input_data:
                    features = test_input_data['features']
                elif 'input_test' in test_input_data and 'features' in test_input_data['input_test']:
                    features = test_input_data['input_test']['features']
                else:
                    st.error("❌ Invalid JSON format")
                    st.session_state["IS_INPUT_AVAILABLE"] = False
                    features = []

                if len(features) == 13:
                    st.session_state["IS_INPUT_AVAILABLE"] = True
                    st.session_state["features"] = features
                else:
                    st.error(f"❌ Expected 13 features, got {len(features)}")
                    st.session_state["IS_INPUT_AVAILABLE"] = False
            else:
                st.session_state["IS_INPUT_AVAILABLE"] = False

        st.markdown("---")
        predict_button = st.button(
            '🔮 Predict Wine Class', type="primary", use_container_width=True)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### 📚 Wine Classes")
        for class_id, info in WINE_DESCRIPTIONS.items():
            st.markdown(f"""
            <div style='padding: 1rem; border-radius: 10px; background: linear-gradient(135deg, {info['color']}22, {info['color']}11); margin: 0.5rem 0; border-left: 4px solid {info['color']}'>
                <h4 style='margin: 0; color: {info['color']}'>{info['emoji']} Class {class_id}: {info['name']}</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666'>{info['characteristics']}</p>
            </div>
            """, unsafe_allow_html=True)

    with col1:
        # Prediction logic
        if predict_button:
            if "IS_INPUT_AVAILABLE" in st.session_state and st.session_state["IS_INPUT_AVAILABLE"]:
                if WINE_MODEL_LOCATION.is_file():
                    try:
                        model = load_model()

                        if model is None:
                            st.error("❌ Failed to load model!")
                            return

                        features = st.session_state["features"]

                        if len(features) != 13:
                            st.error(
                                f"❌ Expected 13 features, got {len(features)}")
                            return

                        with st.spinner('🔬 Analyzing wine properties...'):
                            prediction = model.predict([features])
                            wine_class = int(prediction[0])
                            probabilities = model.predict_proba([features])[0]
                            confidence = probabilities[wine_class] * 100

                        # Beautiful prediction result
                        wine_info = WINE_DESCRIPTIONS[wine_class]
                        st.markdown(f"""
                        <div style='padding: 2rem; border-radius: 20px; background: linear-gradient(135deg, {wine_info['color']}, {wine_info['color']}cc); color: white; text-align: center; margin: 2rem 0; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
                            <h1 style='font-size: 4rem; margin: 0;'>{wine_info['emoji']}</h1>
                            <h2 style='margin: 1rem 0 0.5rem 0;'>Class {wine_class}: {wine_info['name']}</h2>
                            <p style='font-size: 1.5rem; margin: 0; opacity: 0.9;'>Confidence: {confidence:.1f}%</p>
                            <p style='margin: 1rem 0 0 0; opacity: 0.8;'>{wine_info['characteristics']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability chart
                        st.plotly_chart(create_probability_chart(
                            probabilities), use_container_width=True)

                        # Feature visualization
                        st.markdown("### 🎯 Chemical Profile Analysis")
                        st.plotly_chart(create_feature_radar(
                            features, feature_names), use_container_width=True)

                        # Detailed metrics in columns
                        st.markdown("### 📊 Detailed Probabilities")
                        col1, col2, col3 = st.columns(3)

                        for i, col in enumerate([col1, col2, col3]):
                            with col:
                                wine_info = WINE_DESCRIPTIONS[i]
                                delta = f"+{probabilities[i]*100:.1f}%" if i == wine_class else None
                                st.metric(
                                    f"{wine_info['emoji']} Class {i}",
                                    f"{probabilities[i]*100:.1f}%",
                                    delta=delta if i == wine_class else None,
                                    delta_color="normal" if i == wine_class else "off"
                                )

                        # Show input values
                        with st.expander("📋 Input Values Used", expanded=False):
                            input_data = {
                                "Feature": [name.replace('_', ' ').title() for name in feature_names],
                                "Value": [f"{val:.2f}" for val in features]
                            }
                            st.table(input_data)

                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        LOGGER.error(e)
                else:
                    st.error("❌ Model file not found. Run train_wine.py first!")
            else:
                st.warning(
                    "⚠️ Please provide input via sliders or upload a JSON file")
        else:
            # Show welcome message when no prediction yet
            st.markdown("""
            <div style='padding: 3rem; border-radius: 20px; background: linear-gradient(135deg, #667eea22, #764ba222); text-align: center; margin: 2rem 0;'>
                <h2 style='color: #667eea;'>👈 Configure wine features and click Predict</h2>
                <p style='color: #666; font-size: 1.1rem;'>Use the sidebar to input wine properties or upload a JSON file</p>
            </div>
            """, unsafe_allow_html=True)

    # Info section at bottom
    with st.expander("ℹ️ About This Application", expanded=False):
        st.markdown("""
        ### 🎯 How It Works
        
        This AI-powered application uses an **ensemble machine learning model** combining:
        - 🔴 Support Vector Machine (SVM)
        - 🟣 Gradient Boosting
        - 🔵 Random Forest
        
        **Accuracy:** 98-100% on test data
        
        ### 📊 Wine Dataset
        - **178 samples** from three wine cultivars
        - **13 chemical features** analyzed
        - Real data from UCI Machine Learning Repository
        
        ### 🔬 Features Explained
        The model analyzes 13 key chemical properties including alcohol content, acidity levels, 
        phenolic compounds, and color characteristics to accurately classify wines into their 
        respective cultivars.
        ### 🚀 Try It Out!
        """)


if __name__ == "__main__":
    run()
