from flask import Flask, request, jsonify
import pandas as pd
import re
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Configure the Gemini API key from environment variables
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("API key is missing. Set the 'GENAI_API_KEY' environment variable.")
genai.configure(api_key=api_key)

# Define generation configuration
generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}

# Initialize the model
model = genai.GenerativeModel("gemini-pro", generation_config=generation_config)

# Load your database
try:
    file_path = 'Prices_Dimensions_DataCube_07_03_2024.feather'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    products_df = pd.read_feather(file_path)
except Exception as e:
    logging.error(f"Error loading data file: {e}")
    raise

# Normalize the DataFrame for easier matching
products_df['Product_Name'] = products_df['Product_Name'].str.strip()
products_df['Product_Dimension'] = products_df['Product_Dimension'].str.lower().str.strip()
products_df['Product_Type'] = products_df['Product_Type'].str.lower().str.strip()

def analyze_query_with_gemini(query):
    try:
        # Generate a response from Gemini
        response = model.generate_content([f"Analyze the following query and determine if it's related to product information: {query}"])
        
        # Extract the response content
        if response and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = candidate.content.parts[0].text
                return response_text
            else:
                return "No valid content parts found in the response."
        else:
            return "No candidates found in the response."
    
    except Exception as e:
        logging.error(f"Error analyzing query: {e}")
        return "Error analyzing query."

def extract_product_details(query):
    normalized_query = query.lower().strip()
    
    # Use regex patterns to find potential matches for product names and dimensions
    product_name_match = re.search(r'(posteljina\s*moysan\s*satin\s*clorasa|madrac\s*infinity|madrac\s*mondy|posteljina\s*clorasa)', normalized_query)
    dimensions_match = re.search(r'(\d+x\d+)', normalized_query)
    
    product_name = product_name_match.group(0) if product_name_match else None
    dimensions = dimensions_match.group(0) if dimensions_match else None

    logging.info(f"Extracted Product Name: {product_name}")
    logging.info(f"Extracted Dimensions: {dimensions}")

    return product_name, dimensions

def normalize_dimension(dimension):
    if dimension:
        # Standardize dimension to always use the format "width x height"
        parts = dimension.split('x')
        if len(parts) == 2:
            try:
                width = int(parts[0].strip())
                height = int(parts[1].strip())
                return f"{min(width, height)}x{max(width, height)}"
            except ValueError:
                return None
    return None

def validate_dimension(dimension):
    return dimension and re.match(r'^\d+x\d+$', dimension)

def fetch_product_info(product_name, dimensions=None):
    normalized_product_name = product_name.strip() if product_name else None
    dimensions_found = set()
    
    if normalized_product_name:
        if dimensions:
            normalized_dimensions = normalize_dimension(dimensions)
            if not validate_dimension(normalized_dimensions):
                return "Unaprijed navedena dimenzija nije ispravna. Molimo provjerite i unesite valjanu dimenziju."
            
            product_info = products_df[
                (products_df['Product_Name'].str.contains(normalized_product_name, case=False, na=False)) &
                (products_df['Product_Dimension'].apply(lambda x: normalize_dimension(x) == normalized_dimensions))
            ]
            
            if not product_info.empty:
                row = product_info.iloc[0]
                return f"Proizvod {row['Product_Name']} dimenzije {row['Product_Dimension']} košta {row['Product_Price']} KM."
            else:
                # Handle incorrect dimensions
                available_dimensions = set(products_df[
                    products_df['Product_Name'].str.contains(normalized_product_name, case=False, na=False)
                ]['Product_Dimension'])
                return f"Odabrana dimenzija '{dimensions}' nije standardna. Moguće je naručiti dimenziju po vašem izboru. Standardne dimenzije su: {', '.join(available_dimensions)}."
                
        else:
            product_info = products_df[
                products_df['Product_Name'].str.contains(normalized_product_name, case=False, na=False)
            ]
            
            if not product_info.empty:
                dimensions_found = set(product_info['Product_Dimension'])
                if len(dimensions_found) == 1:
                    row = product_info.iloc[0]
                    return f"Proizvod {row['Product_Name']} dimenzije {row['Product_Dimension']} košta {row['Product_Price']} KM."
                elif len(dimensions_found) > 1:
                    return f"Postoji više dimenzija za proizvod '{product_name}'. Molimo vas da navedete dimenziju. Dostupne dimenzije su: {', '.join(dimensions_found)}."
                else:
                    return "Proizvod je pronađen, ali dimenzije nisu dostupne."
            else:
                # Partial match based on product type if no exact match
                product_type_match = products_df[
                    products_df['Product_Type'].str.contains(normalized_product_name.split()[0], case=False, na=False)
                ]
                
                if not product_type_match.empty:
                    # Extract possible matches from product type
                    possible_matches = product_type_match[
                        product_type_match['Product_Name'].str.contains(normalized_product_name.split()[1], case=False, na=False)
                    ]
                    
                    if not possible_matches.empty:
                        row = possible_matches.iloc[0]
                        return f"Proizvod {row['Product_Name']} u tipu {row['Product_Type']} košta {row['Product_Price']} KM."
    
    return "Proizvod nije pronađen."

@app.route('/', methods=['GET'])
def index():
    return "Welcome to my Flask app!"

@app.route('/query', methods=['POST'])
def handle_query():
    query = request.json.get('query', '')
    analysis_result = analyze_query_with_gemini(query)
    logging.info(f"Analysis result: {analysis_result}")

    if "informacije o proizvodu" in analysis_result.lower() or "product information" in analysis_result.lower():
        product_name, dimensions = extract_product_details(query)
        
        if product_name:
            if dimensions:
                # If dimension is provided, normalize and fetch product info directly
                normalized_user_dimension = normalize_dimension(dimensions)
                if not validate_dimension(normalized_user_dimension):
                    return jsonify({"message": "Unaprijed navedena dimenzija nije ispravna. Molimo provjerite i unesite valjanu dimenziju."})
                else:
                    product_info = fetch_product_info(product_name, normalized_user_dimension)
                    if "Odabrana dimenzija" in product_info:
                        # Inform the user about custom solutions if the dimension is not standard
                        return jsonify({"message": product_info})
                    else:
                        return jsonify({"message": f"Product Information: {product_info}"})
            else:
                product_info = fetch_product_info(product_name)
                if "Postoji više dimenzija" in product_info:
                    return jsonify({"message": product_info})
                else:
                    return jsonify({"message": f"Product Information: {product_info}"})
        else:
            return jsonify({"message": "Product name could not be extracted."})
    else:
        return jsonify({"message": "The query is not related to product information."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the PORT environment variable or use 5000 as default
    app.run(host="0.0.0.0", port=port, debug=True) 