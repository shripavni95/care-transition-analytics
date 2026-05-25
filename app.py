import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Care Transition Analytics",
    layout="wide"
)

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("cleaned_dataset.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Dashboard",
        "Analytics",
        "Prediction",
        "Forecasting",
        "Insights"
    ]
)

# -----------------------------------
# DASHBOARD PAGE
# -----------------------------------

if page == "Dashboard":

    st.title("Care Transition Analytics Dashboard")

    st.markdown(
        "### Government Operations Monitoring System"
    )

    # DATE FILTER
    start_date = st.date_input(
        "Start Date",
        df['Date'].min()
    )

    end_date = st.date_input(
        "End Date",
        df['Date'].max()
    )

    filtered_df = df[
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))
    ]

    # KPI CARDS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Apprehended",
        int(filtered_df[
            'Children apprehended and placed in CBP custody*'
        ].sum())
    )

    col2.metric(
        "Total Transfers",
        int(filtered_df[
            'Children transferred out of CBP custody'
        ].sum())
    )

    col3.metric(
        "Total HHS Care",
        int(filtered_df[
            'Children in HHS Care'
        ].sum())
    )

    col4.metric(
        "Total Discharged",
        int(filtered_df[
            'Children discharged from HHS Care'
        ].sum())
    )

    # TREND GRAPH
    st.subheader("Children Apprehended Over Time")

    fig = px.line(
        filtered_df,
        x='Date',
        y='Children apprehended and placed in CBP custody*'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -----------------------------------
# ANALYTICS PAGE
# -----------------------------------

elif page == "Analytics":

    st.title("Advanced Analytics")

    # -----------------------------------
    # HHS CARE TREND
    # -----------------------------------

    st.subheader("HHS Care Trend")

    fig1 = px.line(
        df,
        x='Date',
        y='Children in HHS Care',
        title="Children in HHS Care Over Time"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )
# -----------------------------------
# PREDICTION PAGE
# -----------------------------------

elif page == "Prediction":

    st.title("AI Prediction System")

    st.markdown(
        "### Predict High or Low HHS Care Demand"
    )

    # INPUTS
    apprehended = st.number_input(
        "Children Apprehended",
        min_value=0.0
    )

    custody = st.number_input(
        "Children in CBP Custody",
        min_value=0.0
    )

    transferred = st.number_input(
        "Children Transferred",
        min_value=0.0
    )

    discharged = st.number_input(
        "Children Discharged",
        min_value=0.0
    )

    efficiency = st.number_input(
        "Transfer Efficiency",
        min_value=0.0
    )

    month = st.slider(
        "Month",
        1,
        12,
        1
    )

    day = st.slider(
        "Day",
        1,
        31,
        1
    )

    # PREDICTION
    if st.button("Predict Demand"):

        features = [[
            apprehended,
            custody,
            transferred,
            discharged,
            efficiency,
            month,
            day
        ]]

        prediction = model.predict(features)

        probability = model.predict_proba(features)

        confidence = round(
            probability[0].max() * 100,
            2
        )

        # RESULT
        if prediction[0] == 1:

            st.success(
                f"High HHS Care Demand Predicted "
                f"({confidence}% confidence)"
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence}%"
            )

        else:

            st.warning(
                f"Low HHS Care Demand Predicted "
                f"({confidence}% confidence)"
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence}%"
            )

        # PROBABILITY CHART
        prob_df = pd.DataFrame({
            "Category": [
                "Low Demand",
                "High Demand"
            ],
            "Probability": probability[0]
        })

        fig = px.bar(
            prob_df,
            x='Category',
            y='Probability',
            title="Prediction Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -----------------------------------
# FORECASTING PAGE
# -----------------------------------

elif page == "Forecasting":

    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    st.title("Forecasting Analytics")

    st.markdown(
        "### Future Operational Demand Forecast"
    )

    # Prepare data
    forecast_df = df.copy()

    forecast_df = forecast_df.sort_values("Date")

    series = forecast_df[
        'Children apprehended and placed in CBP custody*'
    ]

    # Forecast Model
    model_forecast = ExponentialSmoothing(
        series,
        trend='add',
        seasonal=None
    ).fit()

    future_predictions = model_forecast.forecast(30)

    # Create dataframe
    future_dates = pd.date_range(
        start=forecast_df['Date'].max(),
        periods=30
    )

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": future_predictions
    })

    # Plot
    fig = px.line(
        future_df,
        x='Date',
        y='Forecast',
        title="30-Day Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(future_df)

    # -----------------------------------
    # TRANSFER EFFICIENCY
    # -----------------------------------

    st.subheader("Transfer Efficiency Distribution")

    fig2 = px.histogram(
        df,
        x='Transfer Efficiency',
        nbins=30,
        title="Operational Efficiency"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # -----------------------------------
    # PIE CHART
    # -----------------------------------

    st.subheader("Operational Distribution")

    totals = {
        "Apprehended":
        df['Children apprehended and placed in CBP custody*'].sum(),

        "Transferred":
        df['Children transferred out of CBP custody'].sum(),

        "Discharged":
        df['Children discharged from HHS Care'].sum()
    }

    pie_df = pd.DataFrame({
        "Category": totals.keys(),
        "Count": totals.values()
    })

    fig3 = px.pie(
        pie_df,
        names='Category',
        values='Count',
        title="Operational Activity Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # -----------------------------------
    # MONTHLY TREND
    # -----------------------------------

    st.subheader("Monthly Average HHS Care")

    monthly = df.groupby('Month')[
        'Children in HHS Care'
    ].mean().reset_index()

    fig4 = px.bar(
        monthly,
        x='Month',
        y='Children in HHS Care',
        title="Average Monthly HHS Care"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )
# -----------------------------------
# INSIGHTS PAGE
# -----------------------------------

elif page == "Insights":

    st.title("Operational Insights")

    st.markdown("""
    ### Key Findings

    - Strong correlation exists between apprehensions and custody counts.
    - Transfer operations directly affect HHS care demand.
    - Operational spikes observed during 2023–2024.
    - Significant decline detected in 2025 operations.
    - Transfer efficiency mostly ranges between 1–2.

    ### Recommendations

    - Improve transfer efficiency processes.
    - Strengthen forecasting systems.
    - Enhance resource allocation planning.
    - Monitor seasonal operational spikes.
    """)