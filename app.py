import streamlit as st
import os
from groq import Groq
import pandas as pd
import numpy as np
import plotly.express as px
from insight_engine import answer_question, generate_insights
from genai_engine import answer_with_ai
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="InsightGen AI",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
[data-testid="stMetric"] {
    background: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
h1 {color: #173b67;}
</style>
""", unsafe_allow_html=True)

st.title("📊 InsightGen AI")
st.subheader("Generative AI-Powered Data Analytics & Decision Support")
st.write(
    "Upload your business dataset to explore performance, "
    "identify data-quality issues, discover insights, and "
    "experiment with machine-learning forecasts."
)

uploaded = st.sidebar.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded is None:
    st.info("Upload a dataset to start your analysis.")
    st.markdown("### What InsightGen AI can do")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**📁 Data profiling**\n\nRows, columns, missing values and duplicates.")
    c2.markdown("**📈 Visual analytics**\n\nDistributions, trends and correlations.")
    c3.markdown("**🤖 Decision support**\n\nRule-based insights and a basic forecast.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

if df.empty:
    st.error("This dataset is empty.")
    st.stop()

# Preserve original data and prepare a working copy.
original_df = df.copy()
df.columns = [str(c).strip() for c in df.columns]

st.sidebar.success("Dataset loaded")
st.sidebar.write(f"File: {uploaded.name}")
st.sidebar.write(f"Rows: {len(df):,}")
st.sidebar.write(f"Columns: {len(df.columns):,}")

# Sidebar data-cleaning options
st.sidebar.markdown("---")
st.sidebar.subheader("Data cleaning")
remove_duplicates = st.sidebar.checkbox("Remove duplicate rows", value=False)
fill_missing = st.sidebar.checkbox("Fill numeric missing values with median", value=False)

if remove_duplicates:
    df = df.drop_duplicates()

if fill_missing:
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_value = df[col].median()
            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)

# Main KPIs
st.markdown("## Dataset overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total rows", f"{len(df):,}")
k2.metric("Total columns", f"{len(df.columns):,}")
k3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
k4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data preview",
    "🔍 Data quality",
    "📊 Visual analytics",
    "💡 Insights & questions",
    "🔮 Forecasting"
])

with tab1:
    st.subheader("Explore your data")
    st.dataframe(df.head(100), use_container_width=True)
    st.caption("Showing up to 100 rows in the preview.")
    st.download_button(
        "Download processed dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="insightgen_processed_data.csv",
        mime="text/csv"
    )

    st.subheader("Statistical summary")
    st.dataframe(df.describe(include="all").transpose(),
                 use_container_width=True)

with tab2:
    st.subheader("Data quality report")
    quality = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data type": df.dtypes.astype(str).values,
        "Missing values": df.isna().sum().values,
        "Missing (%)": (df.isna().mean().values * 100).round(2),
        "Unique values": df.nunique(dropna=True).values
    })
    st.dataframe(quality, use_container_width=True)

    if df.isna().sum().sum() == 0:
        st.success("No missing values detected.")
    else:
        st.warning(
            f"{int(df.isna().sum().sum())} missing cells detected. "
            "Review them before drawing conclusions."
        )

    st.write("**Duplicate rows:**", int(df.duplicated().sum()))

    if df.select_dtypes(include=np.number).empty:
        st.info("No numeric columns are available for outlier screening.")
    else:
        numeric = df.select_dtypes(include=np.number)
        outlier_counts = {}
        for col in numeric.columns:
            q1 = numeric[col].quantile(0.25)
            q3 = numeric[col].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                outlier_counts[col] = 0
            else:
                outlier_counts[col] = int(
                    ((numeric[col] < q1 - 1.5 * iqr) |
                     (numeric[col] > q3 + 1.5 * iqr)).sum()
                )
        st.write("**Potential numeric outliers** (IQR screening)")
        st.dataframe(
            pd.DataFrame(
                list(outlier_counts.items()),
                columns=["Column", "Potential outlier count"]
            ),
            use_container_width=True
        )
        st.caption(
            "Outliers are flags for investigation, not automatically errors."
        )

with tab3:
    st.subheader("Interactive charts")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if numeric_cols:
        selected_num = st.selectbox(
            "Choose a numeric column",
            numeric_cols,
            key="distribution_col"
        )
        fig = px.histogram(
            df, x=selected_num, marginal="box",
            title=f"Distribution of {selected_num}"
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) >= 2:
            x_col = st.selectbox(
                "X-axis", numeric_cols, index=0, key="scatter_x"
            )
            y_col = st.selectbox(
                "Y-axis", numeric_cols,
                index=1, key="scatter_y"
            )
            fig2 = px.scatter(
                df, x=x_col, y=y_col,
                title=f"{y_col} vs {x_col}"
            )
            st.plotly_chart(fig2, use_container_width=True)

        if len(numeric_cols) >= 2:
            st.subheader("Correlation matrix")
            corr = df[numeric_cols].corr()
            fig3 = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                title="Numeric feature correlations",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No numeric columns found for numeric visualizations.")

    if categorical_cols and numeric_cols:
        st.subheader("Compare categories")
        cat_col = st.selectbox("Category", categorical_cols)
        metric_col = st.selectbox(
            "Metric to aggregate", numeric_cols, key="category_metric"
        )
        grouped = (
            df.groupby(cat_col, dropna=False)[metric_col]
              .sum().reset_index()
              .sort_values(metric_col, ascending=False)
              .head(20)
        )
        fig4 = px.bar(
            grouped, x=cat_col, y=metric_col,
            title=f"{metric_col} by {cat_col}"
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.subheader("Automated data insights")

    for insight in generate_insights(df):
        st.write("• " + insight)

    st.caption(
        "Insights are calculated from your uploaded dataset. "
        "Review the results before making business decisions."
    )

    st.subheader("Generative AI Assistant")
    st.write(
        "Ask questions about your uploaded dataset in natural language."
    )

    question = st.text_input(
        "Ask your dataset",
        placeholder="Example: Summarize the main patterns in my data",
        key="genai_dataset_question"
    )

    if st.button("Ask InsightGen AI", key="genai_ask_button"):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("InsightGen AI is analyzing your data..."):
                try:
                    answer = answer_with_ai(df, question.strip())
                    st.markdown("### AI Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"AI error: {type(e).__name__}: {e}")

    st.caption(
        "AI responses are generated from calculated dataset statistics. "
        "Verify important figures before making business decisions."
    )

with tab5:
    st.subheader("Basic machine-learning forecast")
    st.write(
        "This baseline model learns from row order. For meaningful business "
        "forecasting, your dataset should be sorted by time and contain enough "
        "historical observations."
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("Upload a dataset with at least one numeric column.")
    else:
        target = st.selectbox("Select the value to forecast", numeric_cols)
        clean = df[target].dropna().reset_index(drop=True)

        if len(clean) < 5:
            st.warning("At least 5 non-missing observations are required.")
        else:
            test_size = max(1, int(len(clean) * 0.2))
            train_y = clean.iloc[:-test_size]
            test_y = clean.iloc[-test_size:]

            model = LinearRegression()
            X_train = np.arange(len(train_y)).reshape(-1, 1)
            X_test = np.arange(len(train_y), len(clean)).reshape(-1, 1)
            model.fit(X_train, train_y)
            predictions = model.predict(X_test)
            mae = mean_absolute_error(test_y, predictions)

            st.metric("Holdout mean absolute error", f"{mae:,.2f}")
            st.caption(
                "This is a simple baseline, not proof of reliable future accuracy."
            )

            final_model = LinearRegression()
            final_model.fit(
                np.arange(len(clean)).reshape(-1, 1), clean
            )
            horizon = st.slider(
                "Number of future rows to estimate", 1, 12, 3
            )
            future_x = np.arange(
                len(clean), len(clean) + horizon
            ).reshape(-1, 1)
            future_y = final_model.predict(future_x)

            forecast_df = pd.DataFrame({
                "Future row": np.arange(1, horizon + 1),
                f"Predicted {target}": future_y
            })
            st.dataframe(forecast_df, use_container_width=True)

            history = pd.DataFrame({
                "Row": np.arange(1, len(clean) + 1),
                "Actual": clean.values
            })
            future_plot = pd.DataFrame({
                "Row": np.arange(len(clean) + 1, len(clean) + horizon + 1),
                "Actual": future_y
            })
            fig = px.line(
                history, x="Row", y="Actual",
                title=f"Historical {target} and baseline forecast"
            )
            fig.add_scatter(
                x=future_plot["Row"],
                y=future_plot["Actual"],
                mode="lines+markers",
                name="Forecast"
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "InsightGen AI | Student project prototype | "
    "Review results before making business decisions."
)
