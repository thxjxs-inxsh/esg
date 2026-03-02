import os
import ee
import time
from google import genai
from dotenv import load_dotenv
from sec_api import ExtractorApi, QueryApi

# --- INITIALIZATION ---
load_dotenv()

# Required .env variables: 
# SEC_API_KEY, GEMINI_KEY, GEE_PROJECT_ID
SEC_API_KEY = os.getenv("SEC_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID")

# Initialize Clients
# Note: google-genai 2026 SDK uses genai.Client
client = genai.Client(api_key=GEMINI_KEY)
ee.Initialize(project=GEE_PROJECT_ID)

class ESGNormalizer:
    def __init__(self, ticker, lat, lon):
        self.ticker = ticker
        self.coords = ee.Geometry.Point([lon, lat])
        self.extractor = ExtractorApi(SEC_API_KEY)
        self.query_api = QueryApi(SEC_API_KEY)

    def get_sec_esg_score(self, retries=1):
        """Analyzes SEC Item 1 (Business) using Gemini 2.5 Flash-Lite."""
        query = {"query": f"ticker:{self.ticker} AND formType:\"10-K\"", "size": "1"}
        response = self.query_api.get_filings(query)
        
        if not response['filings']:
            return 0.5
        
        # SEC-API 2026 Mapping: use linkToHtml for the main filing body
        filing_url = response['filings'][0]['linkToHtml']
        
        try:
            # Item 1 contains the most proactive ESG strategy text
            content = self.extractor.get_section(filing_url, "1", "text")[:5000]
            
            for i in range(retries + 1):
                try:
                    prompt = (
                        "Task: Act as an ESG Quantitative Auditor. "
                        "Evaluate the following corporate text for environmental commitment. "
                        "Return ONLY a float between 0.0 (no mention/poor) and 1.0 (exemplary). "
                        f"Text: {content}"
                    )
                    
                    # Using Flash-Lite for better 2026 Free Tier RPM (15 requests/min)
                    ai_res = client.models.generate_content(
                        model="gemini-2.5-flash-lite", 
                        contents=prompt
                    )
                    return float(ai_res.text.strip())

                except Exception as e:
                    if "429" in str(e) and i < retries:
                        print(f"⏳ TPM/RPM Limit reached. Sleeping 65s for reset...")
                        time.sleep(65)
                        continue
                    raise e

        except Exception as e:
            print(f"⚠️ AI Analysis Failed ({e}). Falling back to Keyword Density...")
            keywords = ['sustainable', 'carbon', 'renewable', 'emission', 'circular', 'governance']
            matches = sum(content.lower().count(k) for k in keywords)
            return min(matches / 12, 1.0)

    def get_satellite_score(self):
        """Calculates NDVI Ground Truth via GEE."""
        try:
            # Sentinel-2 Harmonized for 2025-2026 data consistency
            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(self.coords)
                  .filterDate('2025-01-01', '2026-12-31')
                  .sort('CLOUDY_PIXEL_PERCENTAGE')
                  .first())
            
            # NDVI Formula: (NIR - Red) / (NIR + Red)
            ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
            stats = ndvi.sample(self.coords, 30).first().get('NDVI').getInfo()
            
            # Normalize [-1.0, 1.0] to [0.0, 1.0]
            return (stats + 1) / 2 if stats is not None else 0.5
        except Exception as e:
            print(f"GEE Error: {e}")
            return 0.5

    def get_audit_commentary(self, final, report, sat):
        """Standard 2026 ESG Benchmarks."""
        # Overall status
        if final > 0.65: status = "ESG LEADER"
        elif final > 0.35: status = "AVERAGE/COMPLIANT"
        else: status = "UNDERPERFORMER/RISK"

        # Integrity Check (Discrepancy between report and ground truth)
        diff = report - sat
        if diff > 0.35:
            integrity = "High probability of Greenwashing (Claims > Reality)."
        elif diff < -0.35:
            integrity = "Hidden Value (Reality > Claims)."
        else:
            integrity = "Claims align with local environmental data."

        return f"{status}: {integrity}"

    def calculate(self):
        report_score = self.get_sec_esg_score()
        sat_score = self.get_satellite_score()
        
        # 2026 Standard Weighting: 60% Corporate Strategy, 40% Local Reality
        final_normalized = (report_score * 0.6) + (sat_score * 0.4)
        
        return {
            "ticker": self.ticker,
            "final_score": round(final_normalized, 3),
            "breakdown": {
                "sentiment": round(report_score, 2),
                "ndvi_reality": round(sat_score, 2)
            },
            "audit": self.get_audit_commentary(final_normalized, report_score, sat_score)
        }

# --- RUN ---
if __name__ == "__main__":
    # Example: Tesla Austin Facility (Giga Texas)
    analyzer = ESGNormalizer("TSLA", 30.222, -97.618)
    results = analyzer.calculate()
    
    print("\n" + "="*40)
    print(f"ESG REPORT: {results['ticker']}")
    print("-" * 40)
    print(f"Normalized Score: {results['final_score']} / 1.0")
    print(f"Report Sentiment: {results['breakdown']['sentiment']}")
    print(f"Satellite NDVI:  {results['breakdown']['ndvi_reality']}")
    print("-" * 40)
    print(f"Auditor Note: {results['audit']}")
    print("="*40)