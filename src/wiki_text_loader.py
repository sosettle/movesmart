import pandas as pd
import numpy as np
import requests
import time
import boto3
import json
import pandas as pd
import time
import os


state_fips_map = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut",
    "10": "Delaware", "11": "District of Columbia",
    "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana",
    "19": "Iowa", "20": "Kansas", "21": "Kentucky",
    "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana",
    "31": "Nebraska", "32": "Nevada", "33": "New Hampshire",
    "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania",
    "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee", "48": "Texas", "49": "Utah",
    "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
    "72": "Puerto Rico"
}


def main():

    df = pd.read_excel("../data/raw/list2_2023.xlsx",
                       sheet_name="List 2", skipfooter=3, header=2)

    df = df.rename(columns={
        'CBSA Code': 'cbsa_code',
        'CBSA Title': 'cbsa_name',
        'Principal City Name': 'principal_city_name',
        'FIPS State Code': 'FIPS_state_code'
    })

    df = df[['principal_city_name', 'cbsa_name', 'cbsa_code', 'FIPS_state_code']]


    def build_wiki_title(row):
        state = state_fips_map[str(row["FIPS_state_code"]).zfill(2)]
        city = row["principal_city_name"]
        return f'{city}, {state}'


    df["wiki_title"] = df.apply(build_wiki_title, axis=1)


    session = requests.Session()
    session.headers.update({
        "User-Agent": "CBSADataProject/1.0 (contact@example.com)"
    })


    def get_wiki_intro(title):

        url = "https://en.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "redirects": 1,
            "titles": title
        }

        try:
            r = session.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()

            pages = data["query"]["pages"]

            for page_id, page in pages.items():
                if page_id == "-1":
                    return None
                return page.get("extract")

        except Exception as e:
            print("API error:", e)
            return None


    texts = []
    for title in df["wiki_title"]:
        text = get_wiki_intro(title)
        if text is None:
            text = ""
        texts.append(text)
        time.sleep(1)

    df["wiki_text"] = texts


    def get_wikivoyage_vibe(city_state):
        url = f"https://en.wikivoyage.org/w/api.php"
        city = city_state.split(",")[0].strip()
        state = city_state.split(",")[1].strip()

        headers = {
            'User-Agent': 'CityRecommenderBot/1.0 (your.email@example.com)'
        }

        attempts = [
            f"{city} ({state})",
            city
        ]

        for title in attempts:
            params = {
                'action': 'query',
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
                'titles': title,
                'format': 'json'
            }

            try:
                response = requests.get(url, params=params,
                                        headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()

                page = next(iter(data['query']['pages'].values()))

                if page.get('pageid', -1) != -1:
                    extract = page.get('extract', '')
                    if extract:
                        return extract

            except requests.exceptions.RequestException as e:
                print(f"Request failed for {title}: {str(e)}")
                continue

        return None


    wikivoyage_texts = []
    for title in df["wiki_title"]:
        text = get_wikivoyage_vibe(title)
        if text is None:
            text = ""
        wikivoyage_texts.append(text)
        time.sleep(1)

    df["wikivoyage_text"] = wikivoyage_texts


    df["city_wiki_wikivoyage_text"] = df["wiki_text"] + "\n" + df["wikivoyage_text"]

    df.to_csv("../data/processed/text/city_wiki_wikivoyage_text_df.csv", index=False)


    df = pd.read_csv("../data/processed/text/city_wiki_wikivoyage_text_df.csv")

    df_cbsa = df.groupby(['cbsa_name', 'cbsa_code'])['city_wiki_wikivoyage_text'] \
        .agg(lambda x: "\n\n".join(x)).reset_index()

    df_cbsa.to_csv("../data/processed/text/cbsa_wiki_wikivoyage_text_df.csv", index=False)


    df_cbsa = pd.read_csv("../data/processed/text/cbsa_wiki_wikivoyage_text_df.csv")


    client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )


    def summarize_cbsa(combined_text, cbsa_name):

         prompt = f"""You are helping someone decide where to relocate in the US.
Pretend you have never heard of these cities before.
You only know what is written in the text below.

Based ONLY on the city descriptions provided below, generate two things:

1. TAGLINE: One short, simple, factual phrase 
capturing the key strengths of {cbsa_name} 
for someone relocating.
Use plain conversational language.
No metaphors, no flowery words, no dramatic tone.
No poetry. Just facts.
Do not include the metro or city name.
Must include at least one specific detail 
unique to this metro.
Based only on provided text.
Example style: "Strong tech job market with 
outdoor lifestyle and mild climate", "Outdoor lifestyle, strong economy, and great access to mountains."

2. SUMMARY: Write a concise, crisp yet comprehensive summary 
in 2-3 short paragraphs of what life is like in the {cbsa_name} 
metro area for someone considering relocating there.
Do not use headers, bullet points, or section titles.
Cover ONLY aspects explicitly mentioned in the text.
If something is not mentioned, skip it entirely.
Do not generate content to fill gaps.
Write in a helpful, conversational tone.
Every sentence must be directly traceable to the text below.
When in doubt, leave it out.

City descriptions:
{combined_text}

You MUST respond in exactly this format:
TAGLINE: [your tagline here]
SUMMARY: [your summary here]
"""

        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "anthropic_version": "bedrock-2023-05-31"
        })

        response = client.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',
            body=body
        )

        result = json.loads(response['body'].read())
        return result['content'][0]['text']


    def parse_llm_output(output):
        try:
            tagline = output.split("TAGLINE:")[1].split("SUMMARY:")[0].strip()
            summary = output.split("SUMMARY:")[1].strip()
            return tagline, summary
        except:
            return None, output


    df_cbsa['tagline'] = None
    df_cbsa['summary'] = None

    for idx, row in df_cbsa.iterrows():

        if pd.notna(row['summary']):
            continue

        try:
            output = summarize_cbsa(
                row['city_wiki_wikivoyage_text'],
                row['cbsa_name']
            )

            tagline, summary = parse_llm_output(output)

            df_cbsa.at[idx, 'tagline'] = tagline
            df_cbsa.at[idx, 'summary'] = summary

            print(f"Done: {row['cbsa_name']}")

            df_cbsa.to_csv('../data/processed/text/cbsa_wiki_wikivoyage_summaries_df.csv', index=False)

            time.sleep(0.5)

        except Exception as e:
            print(f"Failed: {row['cbsa_name']} → {e}")

    print("All done!")


if __name__ == "__main__":
    main()