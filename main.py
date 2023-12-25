import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from math import pi

# Load the pre-trained model and imputer
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
imputer = SimpleImputer(strategy='median')

# Load the dataset
df = pd.read_csv('water.csv')  # Replace 'water_dataset.csv' with your dataset file

# Handle missing values by replacing them with their medians
df.fillna(df.median(), inplace=True)

# Separate features and target variable
X = df.drop('Potability', axis=1)
y = df['Potability']

# Impute missing values with medians
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train the model
rf_model.fit(X_resampled, y_resampled)

# Streamlit App
st.title("Water Potability Prediction")

# Create a form using st.form
with st.form("input_form"):
    st.header("User Input")
    ph = st.text_input("Enter pH:")
    hardness = st.text_input("Enter Hardness:")
    solids = st.text_input("Enter Solids:")
    chloramines = st.text_input("Enter Chloramines:")
    sulfate = st.text_input("Enter Sulfate:")
    conductivity = st.text_input("Enter Conductivity:")
    organic_carbon = st.text_input("Enter Organic Carbon:")
    trihalomethanes = st.text_input("Enter Trihalomethanes:")
    turbidity = st.text_input("Enter Turbidity:")

    # Add a "Predict" button to the form
    predict_button = st.form_submit_button("Predict")

# Process the form only when the "Predict" button is clicked
if predict_button:
    # Validate user input
    try:
        ph = float(ph)
        hardness = float(hardness)
        solids = float(solids)
        chloramines = float(chloramines)
        sulfate = float(sulfate)
        conductivity = float(conductivity)
        organic_carbon = float(organic_carbon)
        trihalomethanes = float(trihalomethanes)
        turbidity = float(turbidity)
    except ValueError:
        st.error("Please enter valid numerical values.")

    if not (0.0 <= ph <= 14.0) or not (47.4 <= hardness <= 323) or not (321 <= solids <= 61200) \
            or not (0.35 <= chloramines <= 13.1) or not (129 <= sulfate <= 481) \
            or not (181 <= conductivity <= 753) or not (2.2 <= organic_carbon <= 28.3) \
            or not (0.74 <= trihalomethanes <= 124) or not (1.45 <= turbidity <= 6.74):
        st.error("Please enter valid values within the specified ranges.")
    else:
        # Make predictions with a loading spinner
        with st.spinner("Predicting..."):
            # Make predictions
            user_data = pd.DataFrame({
                'ph': [ph],
                'Hardness': [hardness],
                'Solids': [solids],
                'Chloramines': [chloramines],
                'Sulfate': [sulfate],
                'Conductivity': [conductivity],
                'Organic_carbon': [organic_carbon],
                'Trihalomethanes': [trihalomethanes],
                'Turbidity': [turbidity]
            })

            # Impute missing values with medians
            user_data_imputed = pd.DataFrame(imputer.transform(user_data), columns=user_data.columns)

            # Ensure the column names match those used during training
            user_data_scaled = scaler.transform(user_data_imputed)

            # Make prediction
            prediction = rf_model.predict(user_data_scaled)

            # Display the prediction result details
            st.subheader("Prediction Result:")
            if prediction[0] == 0:
                st.warning("The water is **Not Drinkable**. It is recommended not to consume.")
            else:
                st.success("Congratulations! The water is **Drinkable**. It is safe for consumption.")
            st.balloons()
            # Additional informative messages
            st.markdown("---")
           

            # Display feature importances
            feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['Importance'])
            feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

            st.subheader("Top Features Contributing to Prediction:")
            st.bar_chart(feature_importances.head(5))  # Display the top 5 features

            # Provide additional text explanations for the top features
            st.write("The top features contributing to the prediction are:")
            for feature, importance in feature_importances.head(5).iterrows():
                st.write(f"- **{feature}:** {importance['Importance']:.4f}")

hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
