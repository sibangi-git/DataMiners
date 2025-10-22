# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import altair as alt

# Title of the Page
st.title("Chronic Disease Screening")

@st.cache_resource
def load_artifacts():
    scaler_heart = joblib.load("models/scaler_XGBoost_logistic_R00.pkl")   
    scaler_diab = joblib.load("models/scaler_hist_dib 2.pkl")
    #model  = joblib.load("models/logistic_model_R00.pkl")   
    diab_model = joblib.load("models/hist_model_dib 1.pkl")
    michd_model = joblib.load("models/logistic_XGBoost_model_R00.pkl")
    return scaler_heart, scaler_diab, diab_model, michd_model

scaler_heart, scaler_diab, diab_model, michd_model = load_artifacts()

# Side Bar 
#with st.sidebar:
#    thresh = st.slider("Decision threshold (for probability models)", 0.0, 1.0, 0.50, 0.01)

with st.sidebar.form("risk_form"):
    sex = st.selectbox("Sex", ["Male", "Female"])
    age_band = st.selectbox("Age group", ["18-24","25-34","35-44","45-54","55-64","65-74"])
    height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.5)
    weight_lbs = st.number_input("Weight (lbs)", min_value=75.0, max_value=1100.0, value=150.0, step=0.5)
    smoker = st.selectbox("Smoking status", ["No","Yes"])
    active = st.selectbox("Any leisure-time physical activity in past 30 days?", ["Yes","No"])
    alcohol_any = st.selectbox("Any alcohol use?", ["Yes","No"])
    ssb_sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
    health_status = st.selectbox("Days physical health not good (past 30 days)", ["0 days", "1–13 days", "14–30 days"])
    edu = st.selectbox("Education level (grouped)", ["<HS","HS/GED","Some college","College+"]) 
    inc = st.selectbox("Income group", ["<15k","15-25k","25-35k","35-50k","50-100k","100-200k","200k+"])  
    submitted = st.form_submit_button("Check my risk")


# Calculations and conversions function
def compute_bmi_and_cat(height_cm: float, weight_lbs: float):
    weight_kg = weight_lbs * 0.45359237
    height_m = height_cm / 100.0
    if not height_m or height_m <= 0:
        return np.nan, np.nan
    bmi = weight_kg / (height_m**2)
    if np.isnan(bmi):
        cat = np.nan
    elif bmi < 18.5:
        cat = 1
    elif bmi >= 18.5 and bmi < 25:
        cat = 2
    elif bmi >= 25 and bmi < 30:
        cat = 3
    else:
        cat = 4
    return float(bmi), cat

def to_model_row(sex, age_band, height_cm, weight_lbs, smoker, active, alcohol_any, ssb_sugar, health_status, edu, inc):
    SEX = 1 if sex == "Male" else 2
    age_map = {"18-24":1,"25-34":2,"35-44":3,"45-54":4,"55-64":5,"65-74":6}
    AGE_G = age_map[age_band]

    bmi, BMI5CAT = compute_bmi_and_cat(height_cm, weight_lbs)

    RFSMOK3 = 1 if smoker == "No" else 2 
    TOTINDA = 1 if active == "Yes" else 2
    DRNKANY6 = 1 if alcohol_any == "Yes" else 2
    ssb_map = {"Low": 1, "Medium": 2, "High": 3}
    SSBSUGR2_CAT = ssb_map[ssb_sugar]
    phys_map = {"0 days":1, "1–13 days":2, "14–30 days":3}
    PHYS14D = phys_map[health_status]

    edu_map = {"<HS":1,"HS/GED":2,"Some college":3,"College+":4}
    inc_map = {"<15k":1,"15-25k":2,"25-35k":3,"35-50k":4,"50-100k":5,"100-200k":6,"200k+":7}
    EDUCAG = edu_map[edu]
    INCOMG1 = inc_map[inc]

    # Build row with names and order your scaler expects
    expected = ['PHYS14D','TOTINDA','SEX','AGE_G','BMI5CAT',
                'EDUCAG','INCOMG1','RFSMOK3','DRNKANY6','SSBSUGR2_CAT']

    # row = {
    #      "SEX": SEX,
    #      "AGE_G": AGE_G,
    #      "BMI5CAT": BMI5CAT,
    #      "RFSMOK3": RFSMOK3,
    #      "TOTINDA": TOTINDA,
    #      "DRNKANY6": DRNKANY6,
    #      "SSBSUGR2": SSBSUGR2_CAT, 
    #      "PHYS14D": PHYS14D,
    #      "EDUCAG": EDUCAG, 
    #      "INCOMG1": INCOMG1
    # }

    row = {
        'PHYS14D': PHYS14D,
        'TOTINDA': TOTINDA,
        'SEX': SEX,
        'AGE_G': AGE_G,
        'BMI5CAT': BMI5CAT,
        'EDUCAG': EDUCAG,
        'INCOMG1': INCOMG1,
        'RFSMOK3': RFSMOK3,
        'DRNKANY6': DRNKANY6,
        'SSBSUGR2_CAT': SSBSUGR2_CAT
    }

    X_raw = pd.DataFrame([row]).reindex(columns=expected)
    #print(X_raw)
    # Ensure numeric dtypes for scaler/model
    X_raw = X_raw.astype(float)

    return X_raw, bmi

# Main Area of the page
col_left, col_right = st.columns(2)

if submitted:
    X_raw, bmi_val = to_model_row(
        sex, age_band, height_cm, weight_lbs, smoker, active, alcohol_any, ssb_sugar, 
        health_status, edu, inc)
    
    expected_heart = ['PHYS14D','TOTINDA','SEX','AGE_G','BMI5CAT',
                'EDUCAG','INCOMG1','RFSMOK3','DRNKANY6','SSBSUGR2_CAT']
    expected_diab = ['PHYS14D','TOTINDA','SEX','AGE_G','BMI5CAT',
                'EDUCAG','INCOMG1','RFSMOK3','DRNKANY6','SSBSUGR2_en']
    X_raw_heart = X_raw.reindex(columns=expected_heart).astype(float)
    X_raw_diab = X_raw.reindex(columns=expected_diab).astype(float)
    X_raw_arth = X_raw.reindex(columns=expected_diab).astype(float)
    # print(X_raw_heart)
    # Transform and predict
    X_heart = scaler_heart.transform(X_raw_heart)  # works if scaler is a fitted transformer or Pipeline
    # print(X_heart)

    # Diabities
    X_diab = scaler_diab.transform(X_raw_diab)  # works if scaler is a fitted transformer or Pipeline
    #print(X_diab)
    probs = diab_model.predict_proba(X_diab)[0]          # e.g., [0.40, 0.45, 0.15]
    classes = list(diab_model.classes_)
    # Map classes to display names
    name_map = {1: "Diabetic", 3: "No diabetes", 4: "Prediabetic"}
    labels = [name_map.get(c, str(c)) for c in classes]

    # Predicted class = argmax
    winner_idx = int(np.argmax(probs))
    print("-------------------------------------------------")
    print(winner_idx)
    winner_name = labels[winner_idx]
    winner_pct = probs[winner_idx] * 100

    if hasattr(michd_model, "predict_proba"):
        p_michd = float(michd_model.predict_proba(X_heart)[0, 1])
        y_michd = int(michd_model.predict(X_heart)[0])
        lab_michd = "Yes" if y_michd == 1 else "No"
    else:
        y_michd = int(michd_model.predict(X_heart)[0])
        p_michd = None
        lab_michd = "Yes" if y_michd == 1 else "No"

    label = "Yes" if y_michd == 1 else "No"
        

    # Left column: echo inputs + BMI and a placeholder chart
    with col_left:
        st.subheader("Patient Information")
        st.markdown(
        f"""
        - Sex: {sex}
        - Age group: {age_band}
        - Height: {height_cm:.1f} cm
        - Weight: {weight_lbs:.1f} lbs
        - Smoking status: {smoker}
        - Active in last 30 days: {active}
        - Any alcohol use: {alcohol_any}
        - Sugar consumption: {ssb_sugar}
        - Physical health not good: {health_status}
        - Education: {edu}
        - Income: {inc}
        """
            )
        if not np.isnan(bmi_val):
            if bmi_val < 18.5:
                bmi_cat = "Underweight"
            elif bmi_val < 25:
                bmi_cat = "Normal"
            elif bmi_val < 30:
                bmi_cat = "Overweight"
            else:
                bmi_cat = "Obesity"
            
            st.markdown( f"#### BMI: {bmi_val:.1f}")
            st.markdown(f"#### Category: {bmi_cat}")
        else:
            st.metric("BMI", "NA")
            st.caption("Category: NA")

        #st.caption("Placeholder chart")  
        #st.line_chart(pd.DataFrame({"Example": np.linspace(0, 1, 20)}))

    # Right column: placeholder predictions until real models are loaded
    with col_right:
        st.subheader("Predicted results")

        st.markdown(f"##### Diabities Classification: {winner_name} ({winner_pct:.1f}%)")
        st.caption("Estimated probabilities from the inputs; not a medical diagnosis.")
        proba_df = pd.DataFrame({
        "Class": labels,
        "Probability": probs,
        "Percent": (probs * 100).round(1)
        })
        proba_df_2 = proba_df[proba_df["Class"].isin(["Diabetic", "Prediabetic"])]

        # Horizontal bar chart with text labels
        bar = alt.Chart(proba_df_2).mark_bar(height=28).encode(
        x=alt.X("Probability:Q", 
                 scale=alt.Scale(domain=[0, 1], nice=False, clamp=True),
                 axis=alt.Axis(format="%", values=[0, 0.25, 0.50, 0.75, 1.00], title="Probability")
        ),
        y=alt.Y(
            "Class:N",
            title=None,
            sort=["Prediabetic", "Diabetic"],
            scale=alt.Scale(paddingInner=0.35, paddingOuter=0.25)
        ),
            color=alt.condition(
            alt.datum.Probability >= 0.50,        
            alt.value("crimson"),                
            alt.value("#1f77b4")           
            )
        ).properties(
            height = alt.Step(34)
        )

        text = alt.Chart(proba_df_2).mark_text(
        align="left", baseline="middle", dx=4, color="black").encode(
        x=alt.X("Probability:Q"),
        y=alt.Y("Class:N", sort="-x"),
        text=alt.Text("Percent:Q", format=".1f")
        )

        st.altair_chart(
            (bar + text).configure_axis(labelOverlap=True),  # extra guard for axis tick overlaps
            use_container_width=True
    )

        if p_michd is not None:
            st.markdown(f"##### Heart Disease: {label}")
            st.caption("Estimated probability of heart attack from the inputs; not a medical diagnosis.")

            # One-bar chart for the positive ("Yes") probability
            heart_df = pd.DataFrame({
                "Class": ["Heart Disease Risk"],
                "Probability": [p_michd],
                "Percent": [round(p_michd * 100, 1)]
            })

            bar = alt.Chart(heart_df).mark_bar().encode(
                x=alt.X("Probability:Q",
                        scale=alt.Scale(domain=[0, 1], nice=False, clamp=True),
                        axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1.0],
                                    title="Probability")),
                y=alt.Y("Class:N", title=None),
                color=alt.condition(
                    alt.datum.Probability >= 0.50,  # red if ≥ 50%
                    alt.value("crimson"),
                    alt.value("#1f77b4")
                )
            ).properties(height=alt.Step(34))

            text = alt.Chart(heart_df).mark_text(
                align="left", baseline="middle", dx=6, color="black"
            ).encode(
                x=alt.X("Probability:Q"),
                y=alt.Y("Class:N"),
                text=alt.Text("Percent:Q", format=".1f")
            )

            st.altair_chart((bar + text).configure_axis(labelOverlap=True), use_container_width=True)
        else:
            st.caption("This model does not output probabilities; showing only the predicted class.")

        st.caption("*Values are model-estimated probabilities based on questionnaire features.")
else:    
    with col_left:
        st.info("Enter details in the sidebar and click 'Check my risk' to see inputs and BMI here.")  
    with col_right:
        st.info("Predicted results will appear here after submission.")
