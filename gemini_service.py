import google.generativeai as genai

def get_ai_insight(api_key, company_name, historical_data, predicted_data):
    try:
        genai.configure(api_key=api_key)
        
        # FIX: Dynamically find an available model to avoid 404
        available_models = [m.name for m in genai.list_models() 
                            if 'generateContent' in m.supported_generation_methods]
        
        # Preference order: 1.5-flash -> 1.5-pro -> gemini-pro -> first available
        if 'models/gemini-1.5-flash' in available_models:
            model_name = 'models/gemini-1.5-flash'
        elif 'models/gemini-1.5-pro' in available_models:
            model_name = 'models/gemini-1.5-pro'
        else:
            model_name = available_models[0]

        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        Analyze the CO2 trend for {company_name}.
        Recent History: {historical_data.tail(3).to_dict()}
        Forecast to 2030: {predicted_data.head(5).to_dict()}
        
        Provide a 3-bullet point summary:
        - Reliability of the trend
        - 2030 Risk Level
        - One engineering recommendation to lower emissions.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Service Error: {str(e)}. Please check your API billing status or key permissions."