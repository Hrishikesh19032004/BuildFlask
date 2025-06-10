from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS
import re
from difflib import SequenceMatcher
import os

app = Flask(__name__)
CORS(app)

# Load data and model once
try:
    data = pd.read_csv('final.csv')
    model = joblib.load('creator_recommendation_model.joblib')
    print("Data and model loaded successfully")
    print(f"Data shape: {data.shape}")
except Exception as e:
    print(f"Error loading data or model: {e}")
    # Create dummy data for testing if files not found
    data = pd.DataFrame()
    model = None

def convert_views_to_number(view_str):
    if isinstance(view_str, str):
        view_str = view_str.strip()
        try:
            if view_str[-1] == 'B':
                return float(view_str[:-1]) * 1e9
            elif view_str[-1] == 'M':
                return float(view_str[:-1]) * 1e6
            elif view_str[-1] == 'K':
                return float(view_str[:-1]) * 1e3
            else:
                return float(view_str.replace(',', ''))
        except Exception:
            return np.nan
    else:
        return view_str

# Apply conversion only if data is loaded
if not data.empty and 'video_views' in data.columns:
    data['video_views'] = data['video_views'].apply(convert_views_to_number)

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_match(target, options, threshold=0.4):
    """Find the best matching option for a target string with lower threshold"""
    best_match = None
    best_score = 0
    
    for option in options:
        score = similarity(target, option)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = option
    
    return best_match, best_score

def find_category_match(target_category, available_categories):
    """Enhanced category matching with multiple strategies"""
    target_lower = target_category.lower()
    
    # Strategy 1: Exact match (case insensitive)
    for cat in available_categories:
        if cat.lower() == target_lower:
            return cat, 1.0
    
    # Strategy 2: Partial match (contains)
    for cat in available_categories:
        if target_lower in cat.lower() or cat.lower() in target_lower:
            return cat, 0.8
    
    # Strategy 3: Fuzzy matching
    best_match, score = find_best_match(target_category, available_categories, threshold=0.3)
    if best_match:
        return best_match, score
    
    # Strategy 4: Category mapping based on keywords
    category_mappings = {
        'sports': ['Sports', 'Fitness', 'Athletics', 'Sport'],
        'fitness': ['Sports', 'Fitness', 'Health'],
        'fashion': ['Fashion', 'Style', 'Clothing'],
        'beauty': ['Beauty', 'Makeup', 'Cosmetics'],
        'tech': ['Technology', 'Tech'],
        'technology': ['Technology', 'Tech'],
        'gaming': ['Gaming', 'Games'],
        'games': ['Gaming', 'Games'],
        'food': ['Food', 'Cooking', 'Recipe'],
        'cooking': ['Food', 'Cooking', 'Recipe'],
        'travel': ['Travel', 'Tourism'],
        'music': ['Music', 'Musical'],
        'education': ['Education', 'Educational'],
        'learning': ['Education', 'Educational'],
        'entertainment': ['Entertainment', 'Comedy'],
        'comedy': ['Entertainment', 'Comedy'],
        'lifestyle': ['Lifestyle', 'Life']
    }
    
    # Check if target matches any keyword
    for keyword, possible_cats in category_mappings.items():
        if keyword in target_lower:
            for possible_cat in possible_cats:
                for available_cat in available_categories:
                    if possible_cat.lower() in available_cat.lower():
                        return available_cat, 0.7
    
    return None, 0

def extract_requirements_from_text(text):
    """Extract requirements from natural language text with improved category matching"""
    text = text.lower()
    
    # Extract budget
    budget = 10000  # default
    budget_patterns = [
        r'budget.*?[\$]?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:k|thousand)?',
        r'[\$]\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:k|thousand)?',
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:k|thousand)?\s*(?:dollars?|usd|\$)',
        r'spend.*?(\d+(?:,\d+)*(?:\.\d+)?)'
    ]
    
    for pattern in budget_patterns:
        match = re.search(pattern, text)
        if match:
            amount = float(match.group(1).replace(',', ''))
            if 'k' in match.group(0) or 'thousand' in match.group(0):
                amount *= 1000
            budget = int(amount)
            break
    
    # Extract number of creators
    creators_count = 5  # default
    creator_patterns = [
        r'(\d+)\s*(?:creators?|influencers?)',
        r'need\s*(\d+)',
        r'find\s*(?:me\s*)?(\d+)',
        r'get\s*(?:me\s*)?(\d+)'
    ]
    
    for pattern in creator_patterns:
        match = re.search(pattern, text)
        if match:
            creators_count = int(match.group(1))
            break
    
    # Extract views requirement
    min_views = 50000  # default
    views_patterns = [
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:k|thousand)?\s*(?:views?|reach)',
        r'minimum.*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:k|thousand)?',
        r'at least.*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:k|thousand)?'
    ]
    
    for pattern in views_patterns:
        match = re.search(pattern, text)
        if match:
            views = float(match.group(1).replace(',', ''))
            if 'k' in match.group(0) or 'thousand' in match.group(0):
                views *= 1000
            min_views = int(views)
            break
    
    # Get available options only if data is loaded
    if not data.empty:
        available_countries = data['country'].dropna().unique()
        available_categories = data['channel_type'].dropna().unique()
    else:
        available_countries = ['United States']
        available_categories = ['Entertainment']
    
    # Extract country with fuzzy matching
    country = "United States"  # default
    country_keywords = {
        'us': 'United States', 'usa': 'United States', 'america': 'United States',
        'united states': 'United States', 'uk': 'United Kingdom', 'britain': 'United Kingdom',
        'england': 'United Kingdom', 'india': 'India', 'canada': 'Canada',
        'australia': 'Australia', 'germany': 'Germany', 'france': 'France',
        'japan': 'Japan', 'brazil': 'Brazil', 'mexico': 'Mexico',
        'italy': 'Italy', 'spain': 'Spain', 'south korea': 'South Korea'
    }
    
    # First check exact keyword matches
    for keyword, country_name in country_keywords.items():
        if keyword in text:
            # Verify this country exists in our data
            best_match, _ = find_best_match(country_name, available_countries)
            if best_match:
                country = best_match
                break
    
    # If no keyword match, try fuzzy matching with available countries
    if country == "United States" and not data.empty:
        words = text.split()
        for word in words:
            if len(word) > 2:  # Skip very short words
                best_match, score = find_best_match(word, available_countries, threshold=0.7)
                if best_match and score > 0.7:
                    country = best_match
                    break
    
    # Extract category with improved matching
    category = "Entertainment"  # default
    
    # Define category keywords more comprehensively
    category_keywords = {
        'sports': 'Sports', 'sport': 'Sports', 'fitness': 'Sports', 'nike': 'Sports', 'adidas': 'Sports',
        'athletic': 'Sports', 'gym': 'Sports', 'workout': 'Sports',
        'fashion': 'Fashion', 'style': 'Fashion', 'clothing': 'Fashion', 'outfit': 'Fashion',
        'beauty': 'Beauty', 'makeup': 'Beauty', 'skincare': 'Beauty', 'cosmetic': 'Beauty',
        'tech': 'Technology', 'technology': 'Technology', 'gadgets': 'Technology', 'software': 'Technology',
        'gaming': 'Gaming', 'games': 'Gaming', 'game': 'Gaming', 'esports': 'Gaming',
        'food': 'Food', 'cooking': 'Food', 'cook': 'Food', 'recipe': 'Food',
        'travel': 'Travel', 'tourism': 'Travel', 'adventure': 'Travel', 'trip': 'Travel',
        'lifestyle': 'Lifestyle', 'vlog': 'Lifestyle', 'daily': 'Lifestyle', 'life': 'Lifestyle',
        'music': 'Music', 'songs': 'Music', 'song': 'Music', 'artist': 'Music',
        'education': 'Education', 'educational': 'Education', 'learning': 'Education', 'tutorial': 'Education',
        'entertainment': 'Entertainment', 'comedy': 'Entertainment', 'funny': 'Entertainment'
    }
    
    # Try keyword matching first
    matched_category = None
    if not data.empty:
        for keyword, category_name in category_keywords.items():
            if keyword in text:
                # Try to find this category in available categories
                matched_cat, score = find_category_match(category_name, available_categories)
                if matched_cat and score > 0.5:
                    matched_category = matched_cat
                    break
        
        # If keyword matching failed, try direct fuzzy matching on the text
        if not matched_category:
            words = text.split()
            for word in words:
                if len(word) > 3:  # Skip very short words
                    matched_cat, score = find_category_match(word, available_categories)
                    if matched_cat and score > 0.6:
                        matched_category = matched_cat
                        break
    
    if matched_category:
        category = matched_category
    
    return {
        'brand_budget_usd': budget,
        'country': country,
        'product_category': category,
        'min_views_required': min_views,
        'creators_count': creators_count
    }

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Creator Recommendation API is running!",
        "endpoints": {
            "/health": "GET - Health check",
            "/recommend": "POST - Get creator recommendations",
            "/categories": "GET - Get available categories",
            "/countries": "GET - Get available countries"
        }
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Check if data is available
        if data.empty:
            return jsonify({
                "error": "Data not available. Please ensure final.csv is uploaded.",
                "top_creators": []
            }), 500
        
        content = request.json
        
        # Check if we have a raw text requirement
        if 'requirements_text' in content:
            parsed_requirements = extract_requirements_from_text(content['requirements_text'])
            brand_budget_usd = parsed_requirements['brand_budget_usd']
            country = parsed_requirements['country']
            product_category = parsed_requirements['product_category']
            min_views_required = parsed_requirements['min_views_required']
            creators_count = parsed_requirements['creators_count']
        else:
            # Use provided structured data
            brand_budget_usd = content.get('brand_budget_usd', 10000)
            country = content.get('country', 'United States')
            product_category = content.get('product_category', 'Entertainment')
            min_views_required = content.get('min_views_required', 50000)
            creators_count = content.get('creators_count', 5)

        # Validate inputs
        if not all([brand_budget_usd, country, product_category, min_views_required]):
            return jsonify({"error": "Missing input parameters"}), 400

        # Get available options for debugging
        available_countries = data['country'].dropna().unique().tolist()
        available_categories = data['channel_type'].dropna().unique().tolist()
        
        print(f"Looking for category: {product_category}")
        print(f"Available categories: {available_categories[:10]}...")
        
        # Improved filtering with better category matching
        # Step 1: Filter by country
        country_filtered = data.copy()
        
        # Try exact country match first
        exact_country_match = country_filtered[country_filtered['country'].str.lower() == country.lower()]
        if not exact_country_match.empty:
            country_filtered = exact_country_match
        else:
            # Try partial country match
            country_matches = country_filtered[country_filtered['country'].str.lower().str.contains(country.lower(), na=False)]
            if not country_matches.empty:
                country_filtered = country_matches
            else:
                # Try fuzzy matching for country
                best_country, score = find_best_match(country, available_countries, threshold=0.6)
                if best_country:
                    country_filtered = country_filtered[country_filtered['country'] == best_country]
                    country = best_country  # Update for response
        
        print(f"After country filtering: {len(country_filtered)} creators")
        
        # Step 2: Filter by category with improved matching
        category_filtered = country_filtered.copy()
        
        # Try exact category match first
        exact_category_match = category_filtered[category_filtered['channel_type'].str.lower() == product_category.lower()]
        if not exact_category_match.empty:
            category_filtered = exact_category_match
            print(f"Exact category match found: {len(category_filtered)} creators")
        else:
            # Try partial category match
            partial_matches = category_filtered[category_filtered['channel_type'].str.lower().str.contains(product_category.lower(), na=False)]
            if not partial_matches.empty:
                category_filtered = partial_matches
                print(f"Partial category match found: {len(category_filtered)} creators")
            else:
                # Try enhanced category matching
                matched_category, score = find_category_match(product_category, available_categories)
                if matched_category and score > 0.4:
                    category_filtered = category_filtered[category_filtered['channel_type'] == matched_category]
                    product_category = matched_category  # Update for response
                    print(f"Enhanced category match '{matched_category}' (score: {score:.2f}): {len(category_filtered)} creators")
                else:
                    print(f"No good category match found for '{product_category}'. Using all categories.")
                    # Don't filter by category if no good match found
        
        filtered = category_filtered.copy()
        
        if filtered.empty:
            return jsonify({
                "top_creators": [], 
                "message": f"No creators found in {country} for {product_category} category. Try different criteria.",
                "parsed_requirements": {
                    "budget": brand_budget_usd,
                    "country": country,
                    "category": product_category,
                    "min_views": min_views_required,
                    "creators_count": creators_count
                },
                "available_options": {
                    "countries": available_countries[:20],
                    "categories": available_categories[:20]
                }
            }), 200

        # Add budget and requirements to the filtered data
        filtered['brand_budget_usd'] = brand_budget_usd
        filtered['min_views_required'] = min_views_required

        # Prepare features for prediction
        features = ['subscribers', 'video_views', 'video_views_for_the_last_30_days']
        
        # Handle missing values
        for feature in features:
            if feature in filtered.columns:
                filtered[feature] = pd.to_numeric(filtered[feature], errors='coerce').fillna(0)
        
        # Create additional features for the model
        filtered['engagement_rate'] = (filtered['video_views_for_the_last_30_days'] / filtered['subscribers']).fillna(0)
        filtered['avg_views_per_video'] = filtered['video_views'].fillna(0)
        
        # Filter by minimum views requirement
        views_filtered = filtered[filtered['video_views'] >= min_views_required]
        
        if views_filtered.empty:
            return jsonify({
                "top_creators": [],
                "message": f"No creators found with minimum {min_views_required:,} views in {country} for {product_category} category. Found {len(filtered)} creators total but none meet view requirements.",
                "parsed_requirements": {
                    "budget": brand_budget_usd,
                    "country": country,
                    "category": product_category,
                    "min_views": min_views_required,
                    "creators_count": creators_count
                }
            }), 200

        # Prepare features for ML model
        try:
            if model is not None:
                X_candidates = views_filtered[features]
                # Predict expected earnings
                predicted_earnings = model.predict(X_candidates)
                views_filtered['predicted_earning'] = predicted_earnings
            else:
                raise Exception("Model not loaded")
        except Exception as e:
            print(f"Model prediction failed: {e}")
            # If model prediction fails, use a simple heuristic
            views_filtered['predicted_earning'] = (
                views_filtered['subscribers'] * 0.001 + 
                views_filtered['video_views'] * 0.0001 + 
                views_filtered['video_views_for_the_last_30_days'] * 0.01
            ) * (brand_budget_usd / 10000)

        # Sort by predicted earnings and get top creators
        top_creators_df = views_filtered.sort_values(by='predicted_earning', ascending=False).head(creators_count)

        # Select and format output fields
        result_columns = ['youtuber', 'predicted_earning', 'subscribers', 'video_views', 'country', 'channel_type']
        result = top_creators_df[result_columns].copy()

        # Ensure all numeric values are properly formatted
        result['predicted_earning'] = result['predicted_earning'].round(2)
        result['subscribers'] = result['subscribers'].fillna(0).astype(int)
        result['video_views'] = result['video_views'].fillna(0).astype(int)

        # Convert to list of dicts
        top_creators = result.to_dict(orient='records')

        return jsonify({
            "top_creators": top_creators,
            "total_found": len(views_filtered),
            "parsed_requirements": {
                "budget": brand_budget_usd,
                "country": country,
                "category": product_category,
                "min_views": min_views_required,
                "creators_count": creators_count
            }
        }), 200

    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "data_shape": data.shape if not data.empty else "No data loaded",
        "model_loaded": model is not None
    }), 200

@app.route('/categories', methods=['GET'])
def get_categories():
    if data.empty:
        return jsonify({"categories": ["Entertainment", "Sports", "Fashion", "Beauty", "Technology"]}), 200
    categories = data['channel_type'].dropna().unique().tolist()
    return jsonify({"categories": categories}), 200

@app.route('/countries', methods=['GET'])
def get_countries():
    if data.empty:
        return jsonify({"countries": ["United States", "India", "United Kingdom", "Canada", "Australia"]}), 200
    countries = data['country'].dropna().unique().tolist()
    return jsonify({"countries": countries}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=False)