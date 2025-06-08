import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

df = pd.read_csv("FINAL_DATASET.csv")

lemmatizer = WordNetLemmatizer()

def filter_food_items(df, diet=None, allergies=None, dislikes=None):
    filtered_df = df.copy()

    #Define common synonyms and variants
    allergen_variants = {
        "dairy": r"\blactose|milk|dairy\b",
        "gluten": r"\bgluten(-free)?\b",
        "soy": r"\bsoy|soya|soja\b",
        "nuts": r"\bnut[s]?\b",
        "egg": r"\begg[s]?\b",
        "shellfish": r"\bshellfish|crustacean[s]?\b",
    }

    def normalize(text):
        return str(text).lower().strip()

    #Diet filtering
    if diet:
        diet = normalize(diet)
        if diet == "pescetarian":
            filtered_df = filtered_df[
                filtered_df["Diet Tags"].str.lower().str.contains("vegetarian", na=False) |
                (filtered_df["Category"].str.lower().str.contains("seafood", na=False))
            ]
        else:
            pattern = allergen_variants.get(diet, fr"\b{re.escape(diet)}\b")
            filtered_df = filtered_df[
                filtered_df["Diet Tags"].str.lower().str.contains(pattern, na=False, regex=True)
            ]

    #Allergy filtering
    if allergies:
        for allergen in allergies:
            allergen = normalize(allergen)
            pattern = allergen_variants.get(allergen, fr"\b{re.escape(allergen)}\b")
            filtered_df = filtered_df[
                ~filtered_df["Allergens"].str.lower().str.contains(pattern, na=False, regex=True)
            ]

    #smart dislike filtering
    if dislikes:
        for dislike in dislikes:
            dislike = normalize(dislike)
            words = dislike.split()
            lemmas = [lemmatizer.lemmatize(word) for word in words]

            def contains_dislike(text):
                text_words = normalize(text).split()
                text_lemmas = [lemmatizer.lemmatize(word) for word in text_words]
                return any(dl in text_lemmas for dl in lemmas)

            filtered_df = filtered_df[
                ~filtered_df["Food Name"].apply(contains_dislike) &
                ~filtered_df["Ingredients"].apply(contains_dislike)
            ]

    return filtered_df


## TEST ##
result = filter_food_items(
    df,
    diet="gluten-free",
    allergies=["dairy", "soya"],
    dislikes=["avocado", "apple"]
)
print(result[["Food Name", "Diet Tags", "Allergens", "Ingredients"]].head())
