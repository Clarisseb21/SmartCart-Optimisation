import pandas as pd
import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from difflib import SequenceMatcher

# Global variable to store user food intentions (set by main optimization script)
CACHED_FOOD_INTENTIONS = {}

def set_food_intentions(intentions_dict):
    """Set the cached food intentions from the main script"""
    global CACHED_FOOD_INTENTIONS
    CACHED_FOOD_INTENTIONS = intentions_dict

def name_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# === Load your files ===
recipes_df = pd.read_csv("recipes_w_search_terms.csv").fillna("").head(20000)
food_df = pd.read_csv("Final _dataset.csv").fillna("")

# === Parse ingredients and steps ===
def safe_eval(text):
    try:
        return eval(text) if text.startswith('[') else [text]
    except:
        return [text]

recipes_df['ingredient_list'] = recipes_df['ingredients'].apply(safe_eval)
recipes_df['steps_list'] = recipes_df['steps'].apply(safe_eval)
recipes_df['steps_list'] = recipes_df['steps_list'].apply(lambda lst: [s.lower() for s in lst if isinstance(s, str)])

# === Normalize for semantic vector use ===
def normalize_food_name(name):
    name = name.lower()
    name = re.sub(r'\b(budget|organic|brand|own|range|label)\b', '', name)
    name = re.sub(r'[^a-z\s]', '', name)
    return '_'.join(name.strip().split())

def extract_brand_type(name):
    name = name.lower()
    if "organic" in name:
        return "organic"
    elif "budget" in name:
        return "budget"
    return "unknown"

# === Build ingredient contexts ===
ingredient_contexts = defaultdict(list)
ingredient_copairs = defaultdict(list)

for _, row in recipes_df.iterrows():
    ingredients = row['ingredient_list']
    steps = row['steps_list']
    clean_ings = ['_'.join(
        re.sub(r'\b(budget|organic|brand|own|range|label)\b', '', re.sub(r'[^\w\s]', '', ing.lower())).strip().split()
        ) for ing in ingredients]

    for ing in clean_ings:
        ing_pattern = re.escape(ing.replace('_', ' '))
        matches = [s for s in steps if re.search(rf'\b{ing_pattern}\b', s)]
        if matches:
            ingredient_contexts[ing].extend(matches)
            ingredient_copairs[ing].extend([i for i in clean_ings if i != ing])

ingredient_contexts = {k: v for k, v in ingredient_contexts.items() if len(v) >= 2}
ingredient_docs = {k: ' '.join(v) for k, v in ingredient_contexts.items()}
ingredient_names = list(ingredient_docs.keys())

# === TF-IDF + Co-occurrence Vectors ===
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_usage = vectorizer.fit_transform(ingredient_docs.values()).toarray()

copair_docs = {k: ' '.join(v) for k, v in ingredient_copairs.items() if k in ingredient_docs}
vectorizer_co = TfidfVectorizer(max_features=1000, stop_words='english')
X_co = vectorizer_co.fit_transform(copair_docs.values()).toarray()

max_dim = max(X_usage.shape[1], X_co.shape[1])
X_usage = np.pad(X_usage, ((0, 0), (0, max_dim - X_usage.shape[1])), 'constant')
X_co = np.pad(X_co, ((0, 0), (0, max_dim - X_co.shape[1])), 'constant')
X_combined = 0.9 * X_usage + 0.1 * X_co

similarity_matrix = cosine_similarity(X_combined)
name_to_index = {name: i for i, name in enumerate(ingredient_names)}

# === Verb Extraction with Lemmatization ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

core_cooking_verbs = {
    "boil", "roast", "grill", "steam", "saute", "bake", "fry", "blanch",
    "cook", "stir", "simmer", "sear", "heat", "mix"
}

def extract_common_verbs(texts):
    verbs = []
    for text in texts:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        for word, tag in tagged:
            if tag.startswith("VB"):
                lemma = lemmatizer.lemmatize(word.lower(), 'v')
                if lemma not in stop_words:
                    verbs.append(lemma)
    return verbs

ingredient_verbs = {k: extract_common_verbs(v) for k, v in ingredient_contexts.items()}

# === Normalize food names ===
food_df['normalized_name'] = food_df['Food Name'].apply(normalize_food_name)
food_df['brand_type'] = food_df['Food Name'].apply(extract_brand_type)

# === Store product mapping ===
ingredient_to_variants = defaultdict(list)
for _, row in food_df.iterrows():
    token = normalize_food_name(row['Food Name'])
    ingredient_to_variants[token].append({
        "store_name": row['Food Name'],
        "category": row['Category'],
        "brand_type": extract_brand_type(row['Food Name']),
        "normalized_name": token,
    })

def get_major_category(category_string):
    """Extract the major food category from a category string"""
    if pd.isna(category_string) or category_string == "":
        return "unknown"
    
    category_lower = str(category_string).lower().strip()
    
    # Define major category mappings
    major_categories = {
        'vegetable': ['vegetable', 'salad', 'cruciferous', 'aromatic'],
        'fruit': ['fruit'],
        'meat': ['meat'],
        'seafood': ['seafood'],
        'dairy': ['dairy', 'yogurt'],
        'pantry': ['pantry'],
        'beverage': ['beverage'],
        'bakery': ['bakery'],
        'snack': ['snack'],
        'condiment': ['condiment'],
        'oil': ['oil'],
        'herb': ['herb'],
        'frozen': ['frozen'],
        'dairy_alternative': ['dairy alternative']
    }
    
    # Check each major category
    for major_cat, keywords in major_categories.items():
        if any(keyword in category_lower for keyword in keywords):
            return major_cat
    
    return "other"

def identify_dual_purpose_foods(food_name):
    """Identify foods that serve multiple purposes"""
    name_lower = str(food_name).lower()
    
    dual_purpose_mapping = {
        'greek yogurt': ['yogurt', 'dairy'],
        'greek yoghurt': ['yogurt', 'dairy'],
        'cottage cheese': ['yogurt', 'dairy'],
        'coconut milk': ['dairy alternative', 'cooking ingredient'],
        'oat milk': ['dairy alternative', 'beverage'],
        'cream cheese': ['dairy', 'condiment'],
    }
    
    for food_key, purposes in dual_purpose_mapping.items():
        if food_key in name_lower:
            return purposes
    
    return None

def get_effective_category(food_name, original_category):
    """Get the effective category using cached intentions - no user prompts"""
    global CACHED_FOOD_INTENTIONS
    
    purposes = identify_dual_purpose_foods(food_name)
    
    if not purposes:
        return original_category
    
    # Use cached intention if available
    food_key = food_name.lower()
    if food_key in CACHED_FOOD_INTENTIONS:
        selected_purpose = CACHED_FOOD_INTENTIONS[food_key]
    else:
        # Fallback to first purpose if not cached (shouldn't happen in normal usage)
        selected_purpose = purposes[0]
    
    # Map purpose to CLEAN category - no mixing!
    purpose_to_category = {
        'yogurt': 'Yogurt',              # Pure yogurt category
        'dairy': 'Dairy',                # Pure dairy category  
        'dairy alternative': 'Dairy Alternative',
        'cooking ingredient': 'Pantry',
        'beverage': 'Beverage',
        'condiment': 'Condiment'
    }
    
    return purpose_to_category.get(selected_purpose, original_category)

def can_suggest_as_alternative(candidate_name, target_category):
    """Check if a dual-purpose food can be suggested for a target category"""
    purposes = identify_dual_purpose_foods(candidate_name)
    if not purposes:
        return False
    
    # Map target categories to purposes
    category_to_purpose = {
        'yogurt': 'yogurt',
        'dairy': 'dairy',
        'dairy alternative': 'dairy alternative',
        'pantry': 'cooking ingredient',
        'beverage': 'beverage',
        'condiment': 'condiment'
    }
    
    target_lower = str(target_category).lower()
    for cat_key, purpose in category_to_purpose.items():
        if cat_key in target_lower and purpose in purposes:
            return True
    
    return False

def are_categories_compatible(cat1, cat2, food1_name="", food2_name=""):
    """Enhanced compatibility check - STRICT matching for clean categories"""
    if pd.isna(cat1) or pd.isna(cat2):
        return False
        
    cat1_lower = str(cat1).lower().strip()
    cat2_lower = str(cat2).lower().strip()
    
    # Exact match is always compatible
    if cat1_lower == cat2_lower:
        return True
    
    # For dual-purpose foods that have been categorized, be VERY strict
    # If Greek yogurt is categorized as "Yogurt", only match with other "Yogurt" items
    clean_categories = ['yogurt', 'dairy', 'dairy alternative', 'pantry', 'beverage', 'condiment']
    
    if any(cat in cat1_lower for cat in clean_categories) and any(cat in cat2_lower for cat in clean_categories):
        # Both are clean categories - must match exactly
        return cat1_lower == cat2_lower
    
    # Check if both contain the same major food type (for non-clean categories)
    major_types = ['vegetable', 'fruit', 'meat', 'seafood']
    for food_type in major_types:
        if food_type in cat1_lower and food_type in cat2_lower:
            return True
    
    # Very limited compatibility for mixed categories
    mixed_compatible_pairs = [
        ('frozen', 'vegetable'),
        ('frozen', 'fruit'),
    ]
    
    for pair in mixed_compatible_pairs:
        if (pair[0] in cat1_lower and pair[1] in cat2_lower) or (pair[1] in cat1_lower and pair[0] in cat2_lower):
            return True
    
    return False

# Modified suggest_store_foods function - no user prompts, uses cached intentions
def suggest_store_foods(ingredient, topn=10):
    query_token = normalize_food_name(ingredient)

    match_row = food_df[food_df['normalized_name'] == query_token]
    if match_row.empty:
        match = get_close_matches(query_token, list(food_df['normalized_name'].unique()), n=1, cutoff=0.6)
        if not match:
            return f"'{ingredient}' not found in store products."
        query_token = match[0]
        match_row = food_df[food_df['normalized_name'] == query_token]

    input_category = match_row.iloc[0]['Category']
    
    # Get effective category using cached intentions (no user prompts)
    effective_category = get_effective_category(ingredient, input_category)
    
    # Filter by compatible categories
    compatible_mask = food_df.apply(lambda row: are_categories_compatible(
        effective_category, 
        row['Category'], 
        ingredient, 
        row['Food Name']
    ), axis=1)
    
    category_rows = food_df[compatible_mask]
    candidate_tokens = category_rows['normalized_name'].unique()

    if query_token in name_to_index:
        idx = name_to_index[query_token]
        sim_scores = similarity_matrix[idx]
    else:
        sim_scores = np.array([0.5] * len(ingredient_names))

    query_verbs = set(ingredient_verbs.get(query_token, []))
    candidates = []

    for token in candidate_tokens:
        raw_sim = 1.0 if token == query_token else (
            sim_scores[name_to_index[token]] if token in name_to_index else 0.5
        )

        candidate_verbs = set(ingredient_verbs.get(token, []))
        shared_verbs = query_verbs & candidate_verbs & core_cooking_verbs
        verb_overlap = len(shared_verbs)

        if raw_sim == 0.5 and verb_overlap == 0:
            fallback_sim = name_similarity(query_token, token) * 0.6
            sim = round(fallback_sim, 4)
        else:
            sim = round(raw_sim, 4)

        verb_score = verb_overlap / max(len(core_cooking_verbs), 1)
        
        # Simplified category bonus - exact match or nothing
        category_bonus = 0.0
        for _, row in food_df[food_df['normalized_name'] == token].iterrows():
            candidate_category = row['Category']
            
            if str(effective_category).lower() == str(candidate_category).lower():
                category_bonus = 0.15  # Exact category match only
            elif are_categories_compatible(effective_category, candidate_category):
                category_bonus = 0.03  # Small bonus for compatible categories
            break
        
        final_score = 0.75 * sim + 0.1 * verb_score + category_bonus

        for _, row in food_df[food_df['normalized_name'] == token].iterrows():
            candidates.append({
                "related_ingredient": token,
                "store_name": row['Food Name'],
                "category": row['Category'],
                "brand_type": row['brand_type'],
                "similarity": sim,
                "verb_overlap": verb_overlap,
                "category_bonus": category_bonus,
                "final_score": round(final_score, 4),
                "shared_verbs": list(shared_verbs)
            })

    top = sorted(candidates, key=lambda x: x['final_score'], reverse=True)[:topn * 2]

    print(f"\n[Search] {ingredient} -> Top related store products (effective category: {effective_category}):\n")
    for item in top:
        print(f"→ {item['store_name']} ({item['brand_type']}, sim={item['similarity']}, verbs={item['verb_overlap']}, cat_bonus={item['category_bonus']:.2f}, match={item['shared_verbs']})")

    return top


### === Example usage ===
#suggest_store_foods("Cucumber - Organic Brand")
#suggest_store_foods("Mince Meat - Budget Brand")
#suggest_store_foods("Olive Oil - Budget Brand")
#suggest_store_foods("Baby Button Mushrooms - Budget Brand")
#suggest_store_foods("Asparagus - Budget Brand")
#suggest_store_foods("Skim Milk")
#suggest_store_foods("Nectarines (1 whole) - Organic Brand")
#suggest_store_foods("Haddock Fillet (200g) - Budget Brand")
#suggest_store_foods("Mustard (500ml) - Budget Brand")
#suggest_store_foods("Greek Yoghurt - Budget Brand")
#suggest_store_foods("Lamb Chops - Budget Brand")
#suggest_store_foods("Tomatoes - Organic Brand")
#suggest_store_foods("Cottage Cheese - Organic Brand")
#suggest_store_foods("Coriander - Organic Brand")
#suggest_store_foods("Artichoke - Budget Brand")
#suggest_store_foods("Grape Juice - Budget Brand")