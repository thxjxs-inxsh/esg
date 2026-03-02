import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_sector_trends(emissions_db, registry):
    """Calculates the average growth/decline for each specific industry sector."""
    # Merge emissions with registry to get 'Sector' info
    df = emissions_db.merge(registry[['Facility Id', 'Industry']], on='Facility Id')
    sector_slopes = {}
    
    for industry in df['Industry'].unique():
        sub = df[df['Industry'] == industry]
        if len(sub) > 5:
            model = LinearRegression().fit(sub[['Year']], sub['Emissions'])
            sector_slopes[industry] = model.coef_[0]
        else:
            sector_slopes[industry] = 0
    return sector_slopes

def predict_for_company(company_hist, industry_name, sector_slopes):
    """Predicts starting from the last available data year using dampened sector trends."""
    if company_hist.empty: return None
    
    last_year = int(company_hist['Year'].max())
    last_val = company_hist['Emissions'].iloc[-1]
    sector_slope = sector_slopes.get(industry_name, 0)
    
    # Calculate company's personal slope
    if len(company_hist) >= 3:
        local_model = LinearRegression().fit(company_hist[['Year']], company_hist['Emissions'])
        local_slope = local_model.coef_[0]
        # Blend: 70% company history, 30% industry sector average
        base_slope = (local_slope * 0.7) + (sector_slope * 0.3)
    else:
        base_slope = sector_slope

    future_years = range(last_year + 1, 2031)
    preds = []
    current_val = last_val
    
    for i, _ in enumerate(future_years):
        # Dampening: The trend slows down by 10% every year (realistic physics)
        dampener = 0.9 ** i 
        current_val += (base_slope * dampener)
        
        # Realism Check: Emissions can't be negative. 
        # Set a floor at 5% of the starting value.
        preds.append(max(last_val * 0.05, current_val))
        
    return pd.DataFrame({'Year': list(future_years), 'Emissions': preds})