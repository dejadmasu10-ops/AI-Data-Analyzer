import pandas as pd


# ── CONFIG ──────────────────────────────────────
DATA_FILE = "sales.csv"
REPORT_FILE = "report.txt"
GEMINI_MODEL = "gemini-3.6-flash"


# ── FUNCTIONS ───────────────────────────────────

def load_data(path):
    """Load the CSV file and return a DataFrame."""
    try:
        return pd.read_csv(path)

    except FileNotFoundError:
        print(f"File not found: {path}")
        return None


def clean_data(df):
    """Remove duplicate rows and missing values."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def analyse(df):
    """Calculate key business insights from the cleaned data."""
    insights = {
        "total_amount": df["amount"].sum(),
        "average_amount": df["amount"].mean(),
        "top_city": df.groupby("city")["amount"].sum().idxmax(),
        "biggest_transaction": df["amount"].max()
    }

    return insights


def ai_summary(insights):
    """Send the insights to Gemini and return an AI-generated report."""
    from google import genai
    from secret import GEMINI_API_KEY

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
    Write a short business report based on these data insights:

    {insights}

    Explain the important findings in simple language.
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text

    except Exception as e:
        print(f"AI error: {e}")
        return None


def save_report(text, path):
    """Save the AI-generated report to a text file."""
    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)

        print(f"Report saved to: {path}")

    except Exception as e:
        print(f"Error saving report: {e}")


def main():
    """Run the complete AI data analysis pipeline."""
    df = load_data(DATA_FILE)

    if df is None:
        return

    df = clean_data(df)

    insights = analyse(df)
    print("Insights:", insights)

    report = ai_summary(insights)

    if report:
        save_report(report, REPORT_FILE)


main()