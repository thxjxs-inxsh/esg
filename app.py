import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predictor import get_sector_trends, predict_for_company
from gemini_service import get_ai_insight
from fpdf import FPDF
import io

# --- CONFIG & THEME ---
st.set_page_config(page_title="EcoCred | Emissions AI", layout="wide", initial_sidebar_state="expanded")

# Toggle this to True if you run out of API quota but want to test the UI
MOCK_MODE = False 

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #10b981; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #059669; border: none; }
    .audit-card { background-color: #1f2937; padding: 20px; border-radius: 10px; border-left: 5px solid #10b981; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# --- PDF GENERATION LOGIC ---
def create_pdf_report(name, industry, current, projection, audit_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "EcoCred Sustainability Audit", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Facility: {name}", ln=True)
    pdf.cell(200, 10, f"Industry: {industry}", ln=True)
    pdf.cell(200, 10, f"Latest Emissions: {current:,.0f} MT CO2e", ln=True)
    pdf.cell(200, 10, f"2030 Projection: {projection:,.0f} MT CO2e", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "AI Analysis:", ln=True)
    pdf.set_font("Arial", '', 11)
    # Handling potential encoding issues for AI text
    clean_text = audit_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_text)
    return pdf.output()

# --- DATA LOADING ---
@st.cache_data
def load_all_data():
    reg = pd.read_csv('data/facility_registry.csv')
    em = pd.read_csv('data/consolidated_emissions.csv')
    slopes = get_sector_trends(em, reg)
    return reg, em, slopes

registry, emissions_db, sector_slopes = load_all_data()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌿 EcoCred")
    st.markdown("---")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google AI Studio Key")
    st.markdown("---")
    st.caption("v2.1.0 - Professional Suite")
    if MOCK_MODE:
        st.warning("Running in Mock Mode (No API used)")

# --- MAIN UI ---
st.title("Emissions Analysis Dashboard")
st.markdown("Search a facility to generate a dampening-linear regression forecast and AI-powered environmental audit.")

# --- UPDATED SEARCH FEATURE ---
# Searchable Selection with a Placeholder
company_name = st.selectbox(
    "Search Facility Registry", 
    options=registry['Facility Name'].unique(),
    index=None,
    placeholder="Type to search (e.g., 'Exxon', 'Power Plant', etc.)..."
)

if company_name:
    facility_info = registry[registry['Facility Name'] == company_name].iloc[0]
    industry = facility_info['Industry']

    if st.button("🚀 Run Comprehensive Audit"):
        hist = emissions_db[emissions_db['Facility Id'] == facility_info['Facility Id']].sort_values('Year')
        
        if not hist.empty:
            with st.spinner("Calculating projections and fetching AI insights..."):
                future_df = predict_for_company(hist, industry, sector_slopes)
                
                # Top Metrics
                m1, m2, m3 = st.columns(3)
                current_val = hist['Emissions'].iloc[-1]
                proj_2030 = future_df['Emissions'].iloc[-1]
                change = ((proj_2030 - current_val) / current_val) * 100
                
                m1.metric("Current Emissions", f"{current_val:,.0f} MT", "Latest Year")
                m2.metric("2030 Projection", f"{proj_2030:,.0f} MT", f"{change:.1f}%")
                m3.metric("Industry Sector", industry[:20], "EPA Category")

                st.markdown("---")

                # Graph and AI Text
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("Emission Trend & Forecast")
                    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0e1117')
                    ax.set_facecolor('#1f2937')
                    
                    # Styling the graph
                    ax.plot(hist['Year'], hist['Emissions'], 'o-', label='Historical', color='#10b981', linewidth=2)
                    
                    # Connection line logic
                    connection = pd.concat([hist.tail(1), future_df.head(1)])
                    ax.plot(connection['Year'], connection['Emissions'], '--', color='#6b7280', alpha=0.5)
                    
                    ax.plot(future_df['Year'], future_df['Emissions'], 'x--', label='AI Forecast', color='#ef4444', linewidth=2)
                    
                    # Grid and Labels
                    ax.tick_params(colors='white')
                    ax.set_ylabel("Metric Tons CO2e", color='white')
                    ax.grid(True, linestyle=':', alpha=0.2)
                    legend = ax.legend(facecolor='#1f2937', edgecolor='#374151')
                    for text in legend.get_texts(): text.set_color("white")
                    st.pyplot(fig)
                    
                with col2:
                    st.subheader("🤖 AI Sustainability Audit")
                    if MOCK_MODE:
                        insight = "MOCK DATA: This facility shows a steady decline in emissions. Recommendation: Invest in carbon capture for stationary combustion units."
                    elif api_key:
                        insight = get_ai_insight(api_key, company_name, hist, future_df)
                    else:
                        insight = "Please provide an API Key to see the AI Audit."
                        st.warning(insight)

                    if (api_key or MOCK_MODE) and 'insight' in locals():
                        st.markdown(f'<div class="audit-card">{insight}</div>', unsafe_allow_html=True)
                        
                        # PDF Download Button
                        try:
                            pdf_bytes = create_pdf_report(company_name, industry, current_val, proj_2030, insight)
                            st.download_button(
                                label="📥 Download Report as PDF",
                                data=pdf_bytes,
                                file_name=f"{company_name}_Audit.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error(f"Could not generate PDF: {e}")

        else:
            st.error("No historical data available for this facility.")
else:
    # This info box shows when the app first loads or when the search is cleared
    st.info("Please search for and select a company above to begin the audit.")
