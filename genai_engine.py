
import json
import pandas as pd
from groq import Groq


def answer_with_ai(df, question):
    if df is None or df.empty:
        return "Please upload a dataset first."

    profile = {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "numeric_statistics": {}
    }

    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if not s.empty:
            profile["numeric_statistics"][str(col)] = {
                "mean": float(s.mean()),
                "max": float(s.max()),
                "min": float(s.min()),
                "sum": float(s.sum())
            }

    import os
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not available to the Streamlit process."
        )
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are InsightGen AI, a data analytics assistant. "
                    "Use the supplied calculated statistics to answer. "
                    "Do not invent numbers. If the profile lacks the "
                    "information, explain that limitation."
                )
            },
            {
                "role": "user",
                "content": (
                    "Dataset statistics:\n"
                    + json.dumps(profile)
                    + "\nQuestion: "
                    + question
                )
            }
        ],
        max_completion_tokens=400
    )

    return response.choices[0].message.content
