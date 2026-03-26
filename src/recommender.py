# recommendations
import pandas as pd

def recommended_cities(
    df,
    user_inputs,
    user_income=None,
    housing_mode="either",
    top_n=10):
    """
    calculate similarity
    """

    ranked = df.copy()  

    if user_income is not None:
        # 30% max is commonly known as the most you can spend on rent
        max_affordable_rent = (user_income/12)*0.30
        # 3-4x user_income is what your house value should be
        max_affordable_home_value = user_income*3.5
    
        if housing_mode in ["rent","either"]:
            ranked = ranked[ranked["MedianGrossRent_B25064"] <= max_affordable_rent]
        
        if housing_mode in ["buy","either"]:
            ranked = ranked[ranked["MedianHomeValue_B25077"] <= max_affordable_home_value]
    
    valid_inputs = {}
    for col, rating in user_inputs.items():
        if col in ranked.columns:
            if rating is not None:
                valid_inputs[col] = rating

    total = 0
    for rating in valid_inputs.values():
        total += rating

    weights = {}
    for col, rating in valid_inputs.items():
        weights[col] = (rating**2)/total

    ranked["recommendation_score"] = 0.0
    for col, weight in weights.items():
        ranked["recommendation_score"] += ranked[col] * weight

    ranked = ranked.sort_values("recommendation_score", ascending=False)

    return ranked.head(top_n)


def cluster_retrieval(cluster_df, scores_df, cbsa_code, top_n=5):
    """
    similar cbsa in cluster
    """
    metro_row = cluster_df[cluster_df["cbsa_code"] == cbsa_code]
    
    if metro_row.empty:
        raise ValueError("CBSA code not found in cluster dataframe.")
        
    cluster_id = metro_row.iloc[0]["hier_cluster"]
    cluster_members = cluster_df[
        cluster_df["hier_cluster"] == cluster_id
    ]["cbsa_code"]

    cluster_members = cluster_members[cluster_members != cbsa_code]
    similar = scores_df[
        scores_df["cbsa_code"].isin(cluster_members)
    ].copy()

    return similar.head(top_n)

def add_text_to_cbsa(df):
    """
    join with text df to get summary
    """
    text_df = pd.read_csv("data/processed/cbsa_wiki_wikivoyage_summaries_df.csv")
    cbsa_codes = list(df['cbsa_code'])
    text_df = text_df.loc[text_df["cbsa_code"].isin(cbsa_codes), :]
    final_df = pd.merge(left=df,right=text_df, on='cbsa_code')
    return final_df
