# Vivid palette — Tableau 10 + modern SaaS accents (colorblind-safe)
COLORS = {
    "churned":    "#E15759",   # vivid coral-red
    "retained":   "#4E79A7",   # strong steel-blue
    "accent":     "#59A14F",   # vibrant green
    "warning":    "#F28E2C",   # bright orange
    "info":       "#76B7B2",   # lively teal
    "purple":     "#B07AA1",   # rich mauve
    "pink":       "#FF9DA7",   # warm pink
    "yellow":     "#EDC949",   # golden yellow
    "brown":      "#9C755F",   # warm brown
    "navy":       "#364F6B",   # deep navy
}

CHURN_COLORS = {"Yes": COLORS["churned"], "No": COLORS["retained"]}
CHURN_COLOR_SEQ = [COLORS["retained"], COLORS["churned"]]

VIVID_CATEGORICAL = [
    "#4E79A7", "#F28E2C", "#E15759", "#76B7B2", "#59A14F",
    "#EDC949", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AB",
]

CUSTOM_CSS = """
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    .metric-card {
        background: linear-gradient(135deg, #4E79A7 0%, #364F6B 100%);
        padding: 1.2rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 6px 20px rgba(78,121,167,0.25);
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; opacity: 0.9; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 2rem; }
    .metric-red {
        background: linear-gradient(135deg, #E15759 0%, #C44E52 100%);
        box-shadow: 0 6px 20px rgba(225,87,89,0.25);
    }
    .metric-green {
        background: linear-gradient(135deg, #59A14F 0%, #76B7B2 100%);
        box-shadow: 0 6px 20px rgba(89,161,79,0.25);
    }
    .metric-orange {
        background: linear-gradient(135deg, #F28E2C 0%, #EDC949 100%);
        box-shadow: 0 6px 20px rgba(242,142,44,0.25);
        color: #1a1a2e;
    }
    .metric-orange h3 { opacity: 0.85; }
    .insight-box {
        background: #f0f4f8; border-left: 4px solid #4E79A7;
        padding: 1rem 1.2rem; border-radius: 0 8px 8px 0;
        margin: 0.8rem 0; font-size: 0.95rem;
        color: #1a1a2e;
    }
    .insight-box b { color: #1a1a2e; }
    .llm-box {
        background: linear-gradient(135deg, #f8f6ff 0%, #ede7f6 100%);
        border-left: 4px solid #B07AA1; padding: 1.2rem;
        border-radius: 0 12px 12px 0; margin: 1rem 0;
        color: #1a1a2e;
    }
    .llm-box b, .llm-box strong, .llm-box h1, .llm-box h2, .llm-box h3, .llm-box h4 { color: #1a1a2e; }
    .llm-box ul, .llm-box ol, .llm-box li, .llm-box p { color: #1a1a2e; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 8px 8px 0 0;
    }
</style>
"""
