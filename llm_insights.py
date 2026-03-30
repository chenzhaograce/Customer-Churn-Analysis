import json
import streamlit as st

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

SYSTEM_PROMPT = """You are a senior data analyst specializing in customer churn analysis.
You provide clear, actionable, and data-driven insights. Use specific numbers from the
data provided. Structure your responses with headers and bullet points for readability.
Be concise but thorough.

IMPORTANT: Do NOT start with preambles like "Of course", "Sure", "Here is",
"Certainly", or any conversational opener. Jump straight into the analysis content.
Output your response in clean HTML format using <h3>, <h4>, <b>, <ul>, <li>, <p> tags.
Do NOT use markdown syntax (no **, ##, or - bullets). Use HTML tags only."""


def _call_openai(api_key: str, prompt: str, max_tokens: int = 1500) -> str:
    if not HAS_OPENAI:
        return "_OpenAI package not installed. Run `pip install openai`._"
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"_OpenAI Error: {str(e)}_"


def _call_gemini(api_key: str, prompt: str, max_tokens: int = 1500) -> str:
    if not HAS_GEMINI:
        return "_google-genai package not installed. Run `pip install google-genai`._"
    try:
        client = genai.Client(api_key=api_key)
        full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{prompt}"
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.4,
            ),
        )
        return response.text
    except Exception as e:
        return f"_Gemini Error: {str(e)}_"


def _call_llm(api_key: str, prompt: str, max_tokens: int = 1500,
              provider: str = "openai") -> str:
    tokens = max_tokens if provider == "openai" else max(max_tokens, 8192)
    if provider == "gemini":
        return _call_gemini(api_key, prompt, tokens)
    return _call_openai(api_key, prompt, tokens)


# ── Generic insight functions ──────────────────────────────────────────────

def get_executive_summary(api_key: str, summary: dict, df_info: str,
                          provider: str = "openai") -> str:
    money_part = ""
    if "money_col" in summary:
        money_part = f"""
- Key Metric Column: {summary['money_col']}
- Avg {summary['money_col']} (Churned): {summary['avg_money_churned']:.2f}
- Avg {summary['money_col']} (Retained): {summary['avg_money_retained']:.2f}
- Total at Risk: {summary['revenue_at_risk']:,.2f}"""

    tenure_part = ""
    if "tenure_col" in summary:
        tenure_part = f"""
- Duration Column: {summary['tenure_col']}
- Avg {summary['tenure_col']} (Churned): {summary['avg_tenure_churned']:.1f}
- Avg {summary['tenure_col']} (Retained): {summary['avg_tenure_retained']:.1f}"""

    prompt = f"""Analyze this customer churn dataset and provide an executive summary.

Dataset Overview:
{df_info}

Key Metrics:
- Total Customers: {summary['total_customers']:,}
- Churned Customers: {summary['churned']:,}
- Churn Rate: {summary['churn_rate']:.1f}%{money_part}{tenure_part}

Provide:
1. Executive Summary - 2-3 sentence overview of the churn situation
2. Key Concerns - Top 3 data-driven concerns
3. Immediate Observations - Notable patterns from the metrics
4. Revenue Impact - Financial implications"""
    return _call_llm(api_key, prompt, max_tokens=2000, provider=provider)


def get_categorical_insights(api_key: str, cat_data: str,
                             provider: str = "openai") -> str:
    prompt = f"""Analyze these categorical feature patterns related to customer churn:

{cat_data}

Provide insights on:
1. Risk Profiles - Which segments churn most?
2. Key Findings - Surprising or notable patterns
3. Targeted Recommendations - Segment-specific retention strategies"""
    return _call_llm(api_key, prompt, max_tokens=2000, provider=provider)


def get_numerical_insights(api_key: str, num_data: str,
                           provider: str = "openai") -> str:
    prompt = f"""Analyze these numerical feature patterns related to customer churn:

{num_data}

Provide insights on:
1. Value Distribution Analysis - How do key metrics differ between churned and retained?
2. Threshold Patterns - Are there critical value thresholds associated with churn?
3. Strategic Recommendations - Data-driven interventions based on these numerical patterns"""
    return _call_llm(api_key, prompt, max_tokens=2000, provider=provider)


def get_model_insights(api_key: str, model_results: dict, feature_importances: dict,
                       feature_names: list, provider: str = "openai") -> str:
    fi_text = ""
    for model_name, imps in feature_importances.items():
        sorted_fi = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)[:10]
        fi_text += f"\n{model_name} Top 10 Features:\n"
        for feat, imp in sorted_fi:
            fi_text += f"  - {feat}: {imp:.4f}\n"

    prompt = f"""Analyze these machine learning model results for customer churn prediction:

Model Performance:
{json.dumps(model_results, indent=2)}

Feature Importances:
{fi_text}

Provide:
1. Model Comparison - Which model performs best and why?
2. Feature Analysis - What do the top features tell us about churn drivers?
3. Model Reliability - Assessment of precision vs recall trade-offs
4. Deployment Recommendations - Which model to use and how to operationalize"""
    return _call_llm(api_key, prompt, max_tokens=3000, provider=provider)


def get_comprehensive_recommendations(api_key: str, summary: dict,
                                      top_churn_factors: str,
                                      provider: str = "openai") -> str:
    prompt = f"""Based on comprehensive analysis of a customer churn dataset:

Churn Overview:
- Churn Rate: {summary['churn_rate']:.1f}%
- Customers at Risk: {summary['churned']:,}

Top Churn Factors:
{top_churn_factors}

Provide a detailed action plan with:
1. Immediate Actions (0-30 days) - Quick wins to reduce churn
2. Short-term Strategy (1-3 months) - Targeted interventions
3. Long-term Strategy (3-12 months) - Structural changes
4. KPIs to Track - Metrics to monitor intervention effectiveness
5. Expected Impact - Estimated churn reduction from each strategy"""
    return _call_llm(api_key, prompt, max_tokens=4096, provider=provider)


def get_segment_deep_dive(api_key: str, segment_data: str,
                          provider: str = "openai") -> str:
    prompt = f"""Perform a deep-dive analysis of this customer segment data:

{segment_data}

Provide:
1. Segment Profiles - Detailed description of high-risk vs low-risk segments
2. Behavioral Patterns - What behaviors precede churn?
3. Intervention Points - Where in the customer journey to intervene
4. Personalization Recommendations - Tailored approaches per segment"""
    return _call_llm(api_key, prompt, max_tokens=3000, provider=provider)
