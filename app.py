import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predictor import get_sector_trends, predict_for_company
from gemini_service import get_ai_insight

# Page Config for a professional feel
st.set_page_config(page_title="EcoTrack AI | Emissions Forecasting", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #10b981; color: white; border: none; }
    .stButton>button:hover { background-color: #059669; border: none; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_all_data():
    reg = pd.read_csv('data/facility_registry.csv')
    em = pd.read_csv('data/consolidated_emissions.csv')
    slopes = get_sector_trends(em, reg)
    return reg, em, slopes

registry, emissions_db, sector_slopes = load_all_data()

# Sidebar Branding
with st.sidebar:
    st.title("🌿 EcoTrack AI")
    st.markdown("---")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google AI Studio Key")
    st.markdown("---")
    st.caption("v2.0.1 - Industry Aware Engine")

# Main Header
st.title("Emissions Analysis Dashboard")
st.markdown("Predicting corporate environmental impact using dampening-linear regression and LLM auditing.")

# Selection Area
col_a, col_b = st.columns([2, 1])
with col_a:
    company_name = st.selectbox("Search Facility Registry", registry['Facility Name'].unique())

facility_info = registry[registry['Facility Name'] == company_name].iloc[0]
industry = facility_info['Industry']

if st.button("🚀 Run Comprehensive Audit"):
    hist = emissions_db[emissions_db['Facility Id'] == facility_info['Facility Id']].sort_values('Year')
    
    if not hist.empty:
        future_df = predict_for_company(hist, industry, sector_slopes)
        
        # Metric Cards
        m1, m2, m3 = st.columns(3)
        current_val = hist['Emissions'].iloc[-1]
        proj_2030 = future_df['Emissions'].iloc[-1]
        change = ((proj_2030 - current_val) / current_val) * 100
        
        m1.metric("Current Emissions", f"{current_val:,.0f} MT", "Latest Year")
        m2.metric("2030 Projection", f"{proj_2030:,.0f} MT", f"{change:.1f}%")
        m3.metric("Industry Sector", industry[:20], "EPA Category")

        st.markdown("---")

        # Visualization and AI Output
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Emission Trend & Forecast")
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0e1117')
            ax.set_facecolor('#1f2937')
            
            # Styling the graph
            ax.plot(hist['Year'], hist['Emissions'], 'o-', label='Historical Data', color='#10b981', linewidth=2)
            connection = pd.concat([hist.tail(1), future_df.head(1)])
            ax.plot(connection['Year'], connection['Emissions'], '--', color='#6b7280', alpha=0.5)
            ax.plot(future_df['Year'], future_df['Emissions'], 'x--', label='AI Forecast', color='#ef4444', linewidth=2)
            
            # Grid and Labels
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.set_ylabel("Metric Tons CO2e")
            ax.grid(True, linestyle=':', alpha=0.2)
            legend = ax.legend(facecolor='#1f2937', edgecolor='#374151')
            for text in legend.get_texts(): text.set_color("white")
            
            st.pyplot(fig)
            
        with col2:
            st.subheader("🤖 AI Sustainability Audit")
            if api_key:
                with st.spinner("Analyzing data patterns..."):
                    insight = get_ai_insight(api_key, company_name, hist, future_df)
                    st.markdown(f"""
                    <div style="background-color: #1f2937; padding: 20px; border-radius: 10px; border-left: 5px solid #10b981;">
                        {insight}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter an API Key in the sidebar to enable AI Audits.")
    else:
        st.error("No historical data available for this facility.")