import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import time

# Initialize session states
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'show_viz' not in st.session_state:
    st.session_state.show_viz = False
if 'show_pred' not in st.session_state:
    st.session_state.show_pred = False

# Main app
st.title("Harare City Council Financial Model")

# File upload section
uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'])
if uploaded_file:
    if st.session_state.df is None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(
                uploaded_file)
            st.session_state.cleaned_df = st.session_state.df.copy()
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# Data cleaning interface
if st.session_state.cleaned_df is not None:
    st.subheader("Data Cleaning")
    with st.expander("Advanced Cleaning Tools"):
        # Numeric missing value handling
        num_cols = st.session_state.cleaned_df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = st.session_state.cleaned_df.select_dtypes(exclude=np.number).columns.tolist()

        st.markdown("**Numerical Columns Handling**")
        num_strategy = st.selectbox(
            "Numerical missing values strategy:",
            ["mean", "median", "mode", "drop"],
            key="num_strat"
        )

        st.markdown("**Categorical Columns Handling**")
        cat_strategy = st.selectbox(
            "Categorical missing values strategy:",
            ["most_frequent", "drop"],
            key="cat_strat"
        )

        if st.button("Apply Cleaning Strategies"):
            try:
                # Handle numerical columns
                if num_strategy != "drop" and num_cols:
                    num_imputer = SimpleImputer(strategy=num_strategy)
                    st.session_state.cleaned_df[num_cols] = num_imputer.fit_transform(
                        st.session_state.cleaned_df[num_cols]
                    )

                # Handle categorical columns
                if cat_strategy != "drop" and cat_cols:
                    cat_imputer = SimpleImputer(strategy=cat_strategy)
                    st.session_state.cleaned_df[cat_cols] = cat_imputer.fit_transform(
                        st.session_state.cleaned_df[cat_cols]
                    )

                # Drop remaining missing values
                st.session_state.cleaned_df.dropna(inplace=True)

                # Convert potential date columns
                date_cols = [col for col in st.session_state.cleaned_df.columns if 'date' in col.lower()]
                for col in date_cols:
                    try:
                        st.session_state.cleaned_df[col] = pd.to_datetime(st.session_state.cleaned_df[col])
                    except Exception as e:
                        pass  # Ignore conversion errors

                st.success("Data cleaning applied successfully!")

            except Exception as e:
                st.error(f"Cleaning error: {str(e)}")

        # Data editing interface
        edited_df = st.data_editor(
            st.session_state.cleaned_df,
            use_container_width=True,
            num_rows="dynamic"
        )
        if st.button("Save Changes"):
            st.session_state.cleaned_df = edited_df
            st.success("Data changes saved!")

# Visualization section
if st.session_state.cleaned_df is not None:
    st.subheader("Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Toggle Visualization"):
            st.session_state.show_viz = not st.session_state.show_viz

    with col2:
        if st.button("🔮 Toggle Prediction"):
            st.session_state.show_pred = not st.session_state.show_pred

    # Visualization logic
    if st.session_state.show_viz:
        viz_type = st.selectbox(
            "Choose Visualization Type",
            ["Line Chart", "Bar Chart", "Pie Chart", "Histogram"],
            key="viz_type"
        )

        try:
            all_columns = st.session_state.cleaned_df.columns.tolist()
            if viz_type == "Line Chart":
                x_axis = st.selectbox("X-axis", options=all_columns)
                y_axis = st.selectbox("Y-axis", options=all_columns)
                fig = px.line(st.session_state.cleaned_df, x=x_axis, y=y_axis)

            elif viz_type == "Bar Chart":
                x_axis = st.selectbox("X-axis", options=all_columns)
                y_axis = st.selectbox("Y-axis", options=all_columns)
                fig = px.bar(st.session_state.cleaned_df, x=x_axis, y=y_axis)

            elif viz_type == "Pie Chart":
                category = st.selectbox("Category", options=all_columns)
                value = st.selectbox("Value", options=all_columns)
                fig = px.pie(st.session_state.cleaned_df, names=category, values=value)

            elif viz_type == "Histogram":
                column = st.selectbox("Select Column", options=all_columns)
                fig = px.histogram(st.session_state.cleaned_df, x=column)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Visualization error: {str(e)}")

    # Prediction logic with real-time fluctuations
    if st.session_state.show_pred:
        try:
            # Ensure the date column is not dropped
            date_cols = [col for col in st.session_state.cleaned_df.columns if 'date' in col.lower()]

            # Convert all features to numeric, excluding date columns
            non_date_cols = [col for col in st.session_state.cleaned_df.columns if col not in date_cols]
            for col in non_date_cols:
                st.session_state.cleaned_df[col] = pd.to_numeric(st.session_state.cleaned_df[col], errors='coerce')

            # Drop columns that couldn't be converted
            bad_cols = st.session_state.cleaned_df.columns[
                st.session_state.cleaned_df.isna().all()
            ]
            if not bad_cols.empty:
                st.warning(f"Removed non-convertible columns: {', '.join(bad_cols)}")
                st.session_state.cleaned_df.drop(columns=bad_cols, inplace=True)

            target = st.selectbox("Select Target Variable",
                                  st.session_state.cleaned_df.columns)
            features = st.multiselect("Select Features",
                                      [col for col in st.session_state.cleaned_df.columns if col != target])

            if features and target:
                # Ensure valid data types
                X = st.session_state.cleaned_df[features].astype(np.float64)
                y = st.session_state.cleaned_df[target].astype(np.float64)

                # Final check for remaining missing values
                if X.isnull().any().any() or y.isnull().any():
                    st.error("Missing values detected after conversion. Please clean data first.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    model = RandomForestRegressor()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    # Predict next 12 months with fluctuations
                    future_X = pd.DataFrame(np.zeros((12, len(features))), columns=features)
                    future_X.iloc[:, :] = X.iloc[-1].values  # Use last row as baseline
                    future_predictions = model.predict(future_X)

                    # Add fluctuations to future predictions
                    fluctuations = np.random.normal(0, 10, size=12)  # Random fluctuations
                    future_predictions_with_fluctuations = future_predictions + fluctuations

                    # Prediction visualization
                    viz_type = st.selectbox(
                        "Prediction Display",
                        ["Actual vs Predicted", "Residual Plot", "Feature Importance", "Future Predictions"],
                        key="pred_viz"
                    )

                    if viz_type == "Actual vs Predicted":
                        fig = px.scatter(x=y_test, y=predictions,
                                         labels={'x': 'Actual', 'y': 'Predicted'})
                        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                                      x1=y_test.max(), y1=y_test.max())
                        st.plotly_chart(fig)

                    elif viz_type == "Residual Plot":
                        residuals = y_test - predictions
                        fig = px.scatter(x=y_test, y=residuals)
                        st.plotly_chart(fig)

                    elif viz_type == "Feature Importance":
                        importances = model.feature_importances_
                        fig = px.bar(x=features, y=importances)
                        st.plotly_chart(fig)

                    elif viz_type == "Future Predictions":
                        future_months = np.arange(1, 13)
                        fig = px.line(x=future_months, y=future_predictions_with_fluctuations)
                        st.plotly_chart(fig)

                    st.metric("Mean Absolute Error", f"USD {mean_absolute_error(y_test, predictions):,.2f}")

            # Additional visualization options for future predictions
            future_viz_type = st.selectbox(
                "Future Predictions Visualization",
                ["Line Graph", "Bar Chart", "Histogram"],
                key="future_viz"
            )

            if features and target and 'future_predictions_with_fluctuations' in locals():
                if future_viz_type == "Line Graph":
                    future_months = np.arange(1, 13)
                    fig = px.line(x=future_months, y=future_predictions_with_fluctuations)
                    st.plotly_chart(fig)

                elif future_viz_type == "Bar Chart":
                    future_months = np.arange(1, 13)
                    fig = px.bar(x=future_months, y=future_predictions_with_fluctuations)
                    st.plotly_chart(fig)

                elif future_viz_type == "Histogram":
                    fig = px.histogram(x=future_predictions_with_fluctuations)
                    st.plotly_chart(fig)
            else:
                st.info("Please ensure data is properly cleaned and features are selected to view future predictions.")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
