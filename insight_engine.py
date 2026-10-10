import pandas as pd
import numpy as np


def answer_question(df, question):
    """Answer supported questions using actual dataset calculations."""
    q = question.lower().strip()

    if not q:
        return "Please enter a question."

    if any(x in q for x in [
        "how many rows", "row count", "number of rows"
    ]):
        return f"Your dataset contains {len(df):,} rows."

    if any(x in q for x in [
        "how many columns", "column count", "number of columns"
    ]):
        return f"Your dataset contains {len(df.columns):,} columns."

    if "missing values" in q:
        return (
            f"Your dataset contains "
            f"{int(df.isna().sum().sum()):,} missing cells."
        )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Identify a column explicitly mentioned in the question.
    matched_col = next(
        (c for c in df.columns if str(c).lower() in q),
        None
    )

    if matched_col is None:
        return (
            "I couldn't identify the column. Include its exact name, "
            "for example: What is the average Sales?"
        )

    if matched_col not in numeric_cols:
        if any(w in q for w in ["unique", "categories", "different values"]):
            values = df[matched_col].dropna().astype(str).unique()[:20]
            return f"Values in {matched_col}: {', '.join(values)}"
        return f"{matched_col} is not numeric. Try asking for its unique values."

    series = df[matched_col].dropna()

    if series.empty:
        return f"There are no numeric values to analyze in {matched_col}."

    if any(w in q for w in ["average", "mean", "avg"]):
        return f"Average {matched_col}: {series.mean():,.2f}"

    if any(w in q for w in ["highest", "maximum", "largest", "max"]):
        return f"Highest {matched_col}: {series.max():,.2f}"

    if any(w in q for w in ["lowest", "minimum", "smallest", "min"]):
        return f"Lowest {matched_col}: {series.min():,.2f}"

    if any(w in q for w in ["total", "sum"]):
        return f"Total {matched_col}: {series.sum():,.2f}"

    if "median" in q:
        return f"Median {matched_col}: {series.median():,.2f}"

    if "standard deviation" in q or "std" in q:
        return f"Standard deviation of {matched_col}: {series.std():,.2f}"

    return (
        "I found the column, but couldn't identify the calculation. "
        "Try average, total, median, highest, lowest, or standard deviation."
    )


def generate_insights(df):
    """Generate observations from computed statistics."""
    insights = [
        f"Dataset size: {len(df):,} rows and {len(df.columns):,} columns.",
        f"Missing cells: {int(df.isna().sum().sum()):,}.",
        f"Duplicate rows: {int(df.duplicated().sum()):,}."
    ]

    numeric = df.select_dtypes(include=np.number)

    for col in numeric.columns[:5]:
        values = numeric[col].dropna()

        if not values.empty:
            insights.append(
                f"{col}: average {values.mean():,.2f}; "
                f"median {values.median():,.2f}; "
                f"minimum {values.min():,.2f}; "
                f"maximum {values.max():,.2f}."
            )

    return insights
