import os
import json
import ee
import requests
import pandas as pd
from geopy.geocoders import Nominatim
from google import genai
from google.genai import types
from sec_api import QueryApi, ExtractorApi

# --- 1. CONFIGURATION & CREDIT SHIELD ---
SEC_CACHE_FILE = "sec_cache.json"

def load_sec_cache():
    if os.path.exists(SEC_CACHE_FILE):
        with open(SEC_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_sec_cache(cache):
    with open(SEC_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

# --- 2. INITIALIZATION ---
try:
    ee.Initialize(project='project-id')
except:
    print("GEE Initialization failed. Run 'earthengine authenticate'.")

# API Keys
SEC_API_KEY = "SEC API KEY"
GEMINI_KEY = "Gemini"

client = genai.Client(api_key=GEMINI_KEY)
queryApi = QueryApi(api_key=SEC_API_KEY)
extractorApi = ExtractorApi(api_key=SEC_API_KEY)
geolocator = Nominatim(user_agent="orbit_audit_api_v7")

# --- 3. CORE PROCESSING ENGINE ---

def run_forensic_audit(company_name, ticker):
    """
    Runs the full pipeline and returns a structured JSON object.
    Includes Geocoding, Satellite Multi-Gas, SEC 10-K, and AI Analysis.
    """
    
    # A. GEOLOCATION
    search_query = f"{company_name} refinery plant"
    location = geolocator.geocode(search_query, timeout=10)
    if not location:
        return {"status": "error", "message": "Industrial site not found."}
    
    lat, lon = location.latitude, location.longitude
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(1200).bounds()

    # B. SEC DATA (CREDIT-PROTECTED)
    cache = load_sec_cache()
    if ticker in cache:
        sec_evidence = cache[ticker]
    else:
        try:
            query = {"query": f"ticker:{ticker} AND formType:\"10-K\"", "from": "0", "size": "1"}
            filings = queryApi.get_filings(query)
            if filings and filings['filings']:
                f = filings['filings'][0]
                url = f.get('linkToFilingDetails') or f.get('filingUrl')
                # Extract Item 1A (Risk Factors)
                raw_text = extractorApi.get_section(url, "1A", "text")
                sec_evidence = {
                    "year": f.get('filedAt')[:4],
                    "text": raw_text[:3000] if raw_text else "No sustainability text found.",
                    "url": url
                }
                cache[ticker] = sec_evidence
                save_sec_cache(cache)
            else:
                sec_evidence = {"year": "N/A", "text": "No filing found.", "url": ""}
        except:
            sec_evidence = {"year": "N/A", "text": "SEC API Error.", "url": ""}

    # C. SATELLITE MULTI-GAS & VISUALS
    atmospheric_results = []
    visual_assets = {}
    years = [2022, 2024]
    
    # Map Vis: 50% opacity to see factory infrastructure
    no2_vis = {'min': 0, 'max': 0.00018, 'palette': ['blue', 'purple', 'red', 'yellow'], 'opacity': 0.5}

    for year in years:
        start, end = f"{year}-01-01", f"{year}-12-31"
        
        # Datasets
        no2_img = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2').select('tropospheric_NO2_column_number_density').filterBounds(region).filterDate(start, end).mean()
        ch4_img = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4').select('CH4_column_volume_mixing_ratio_dry_air').filterBounds(region).filterDate(start, end).mean()
        s2_base = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filterDate(start, end).sort('CLOUDY_PIXEL_PERCENTAGE').first()

        # Data Points
        no2_val = no2_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=1000).getInfo().get('tropospheric_NO2_column_number_density', 0)
        ch4_val = ch4_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=7000).getInfo().get('CH4_column_volume_mixing_ratio_dry_air', 0)

        # Append Gas Data
        atmospheric_results.append({
            "year": year,
            "methane_ppb": round(ch4_val or 0, 1),
            "no2_density": round(no2_val or 0, 8),
            "calculated_co2_tonnes": round((no2_val or 0) * 250000, 2)
        })

        # Generate Visual Heatmap PNG URL
        blended = s2_base.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000).blend(no2_img.visualize(**no2_vis))
        visual_assets[year] = blended.getThumbURL({'dimensions': 800, 'region': region, 'format': 'png'})

    # D. AI FORENSIC ANALYSIS
    analysis_prompt = (
        f"Sustainability Audit for {company_name}.\n"
        f"SEC Disclosure ({sec_evidence['year']}): {sec_evidence['text'][:1500]}\n"
        f"Atmospheric Reality (2022-2024): {atmospheric_results}\n"
        "Analyze the gap between legal claims and gas measurements. Provide a Trust Score 1-100."
    )
    
    # Vision analysis (fetching thumbnails for Gemini)
    vision_parts = [analysis_prompt]
    for year in years:
        try:
            img_data = requests.get(visual_assets[year]).content
            vision_parts.append(types.Part.from_bytes(data=img_data, mime_type="image/png"))
        except: pass

    try:
        ai_response = client.models.generate_content(model="gemini-3-flash-preview", contents=vision_parts)
        verdict = ai_response.text
    except:
        verdict = "AI analysis failed, refer to raw atmospheric data."

    # E. FINAL JSON CONSTRUCTION
    master_json = {
        "status": "success",
        "metadata": {
            "company": company_name,
            "ticker": ticker,
            "site_address": location.address,
            "coordinates": {"lat": lat, "lon": lon}
        },
        "audit_assets": {
            "heatmaps": visual_assets,
            "gas_metrics": atmospheric_results
        },
        "legal_evidence": sec_evidence,
        "ai_verdict": {
            "report": verdict,
            "timestamp": "2026-03-02T16:55:00Z"
        }
    }
    
    return master_json

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # Test with a major industrial site
    result = run_forensic_audit("Exxon Mobil Baytown", "XOM")
    
    # Print the JSON output
    print(json.dumps(result, indent=4))
    
    # Optional: Write to a file
    with open("audit_result.json", "w") as outfile:

        json.dump(result, outfile, indent=4)
