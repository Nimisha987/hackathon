    # chatbot.py
    import google.generativeai as genai
    import os
    import requests

    # --- API Key Configuration ---

    # Correctly load Gemini API key from environment variable
    # os.getenv() expects the variable name as a string (e.g., "GEMINI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        # It's better to raise an error early if the key is missing during development/startup
        # In a production app, you might log this and exit gracefully.
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it.")

    # Correctly configure the genai library with the Gemini API key
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

    # Correctly load Google Maps API key from environment variable
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not google_maps_api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set. Please set it.")

    # --- End API Key Configuration ---


    def get_nearby_doctors(specialist_type, location, api_key):
        """
        Fetches nearby doctors of a specified type based on location using Google Places API.
        Args:
            specialist_type (str): The type of medical specialist (e.g., "dermatologist", "general physician").
            location (str): The user's current location (e.g., "Delhi, India").
            api_key (str): Your Google Maps Places API key.
        Returns:
            str: A formatted string of nearby doctor suggestions or a "not found" message.
        """
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        query = f"{specialist_type} near {location}"
        
        params = {
            "query": query,
            "key": api_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            results = response.json().get("results", [])

            if not results:
                return f"No nearby {specialist_type}s found in {location}."

            suggestions = []
            for place in results[:5]:  # Limit to top 5 results
                name = place.get("name")
                address = place.get("formatted_address")
                rating = place.get("rating", "No rating")
                suggestions.append(f"{name}, {address} – ⭐ {rating}")

            return "\n".join(suggestions)

        except requests.exceptions.RequestException as e:
            return f"Error fetching nearby doctors: {e}. Please check your internet connection or API key."
        except Exception as e:
            return f"An unexpected error occurred while processing doctor suggestions: {e}"


    def get_disease_info(symptom_text, image_disease, user_location):
        """
        Generates information about a disease/condition and suggests nearby doctors
        using the Gemini API and Google Maps API.
        Args:
            symptom_text (str): Symptoms provided by the user (if any).
            image_disease (str): Disease label suggested by an image classifier (if any).
            user_location (str): The user's current location for doctor suggestions.
        Returns:
            str: A comprehensive response including disease info and nearby doctor suggestions.
        """
        # Construct a prompt for the Gemini model based on available information
        prompt = "You are a helpful AI assistant for health information.\n"
        if symptom_text:
            prompt += f"The user has the following symptoms: {symptom_text}.\n"
        if image_disease:
            prompt += f"An image classifier suggests the condition: {image_disease}.\n"
        
        prompt += """
        Explain:
        1. What the problem/condition is (in simple language).
        2. How serious it is (e.g., mild, moderate, severe, urgent).
        3. What type of doctor to consult for this (e.g., General Physician, Dermatologist, Specialist, Emergency Room).
        Give a concise and informative response.
        """

        try:
            # Generate content using the Gemini model
            response = model.generate_content(prompt)
            answer = response.text

            # --- Rule-based extraction of specialist type for Google Maps API ---
            specialist = "doctor" # Default specialist
            
            # Check for keywords in the generated answer to determine specialist
            if "dermatologist" in answer.lower():
                specialist = "dermatologist"
            elif "gastroenterologist" in answer.lower():
                specialist = "gastroenterologist"
            elif "general physician" in answer.lower() or "general practitioner" in answer.lower():
                specialist = "general physician"
            elif "cardiologist" in answer.lower():
                specialist = "cardiologist"
            elif "pediatrician" in answer.lower():
                specialist = "pediatrician"
            elif "emergency room" in answer.lower() or "hospital" in answer.lower():
                specialist = "hospital"
            # Add more rules for other specialists as needed

            # Suggest doctors using Google Maps API
            # Pass the environment-variable-loaded API key
            nearby_doctors = get_nearby_doctors(specialist, user_location, google_maps_api_key)

            return f"{answer}\n\n--- Nearby Medical Professionals ---\n{nearby_doctors}"

        except Exception as e:
            return f"An error occurred while generating disease information: {e}. Please try again later."
    