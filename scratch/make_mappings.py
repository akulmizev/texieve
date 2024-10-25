import json
import pandas as pd

from collections import defaultdict

from langcodes import *

from nllb_lang_pairs import NLLB_PAIRS, CCMATRIX_PAIRS

TO_REMOVE = ["Hant", "Hans", "Japn", "Cyrs"]


df = pd.read_csv("GlotScript.tsv", na_filter=False, sep="\t")
scripts = {
    row["ISO639-3"]: row["ISO15924-Main"].split(", ") for _, row in df.iterrows()
}

with open("lang_mappings.json", "r") as f:
    wiki_mappings = json.load(f)

with open("bible_langs.txt", "r") as f:
    bible_langs = [lang.strip() for lang in f.read().split(",")]

with open("mC4_langs.tsv", "r") as f:
    mc4_langs_raw = [
        line.split("\t")[0]
        for line in f.read().split("\n")
        if line.split("\t")[0].isalpha()
    ]
    mc4_langs_raw.remove("iw")
    mc4_langs_raw.remove("la")
    mc4_langs_raw.remove("und")
    mc4_langs_raw.append("he")
    mc4_langs = [Language.get(lang).to_alpha3() for lang in mc4_langs_raw]

nllb_mappings = defaultdict(list)
for pair in NLLB_PAIRS:
    lang_1 = pair[0].split("_")[0]
    lang_2 = pair[1].split("_")[0]
    config_str = "-".join(pair)
    nllb_mappings[lang_1].append(config_str)
    nllb_mappings[lang_2].append(config_str)

for pair in CCMATRIX_PAIRS:
    lang_1 = pair[0].split("_")[0]
    lang_2 = pair[1].split("_")[0]
    config_str = "-".join(pair)
    nllb_mappings[lang_1].append(config_str)
    nllb_mappings[lang_2].append(config_str)


all_langs = (
    set(wiki_mappings.keys())
    | set(bible_langs)
    | set(mc4_langs)
    | set(nllb_mappings.keys())
)
final_mappings = {}

for lang in all_langs:
    sources = []
    lang_scripts = scripts.get(lang, [])
    wikipedia_id = None
    lang_metadata = Language.get(lang)
    if lang in wiki_mappings.keys():
        sources.append("wiki")
        wikipedia_id = wiki_mappings[lang]["wiki_id"]
        lang_scripts.extend(wiki_mappings[lang]["scripts"])
    if lang in bible_langs:
        sources.append("bible")
    if lang in mc4_langs:
        sources.append("mc4")
    if lang in nllb_mappings:
        sources.append("nllb")
    lang_scripts = [
        script for script in list(set(lang_scripts)) if script not in TO_REMOVE
    ]
    mappings = {
        "bcp_47_code": standardize_tag(lang),
        "wikipedia_id": wikipedia_id,
        "language": lang_metadata.language_name(),
        "scripts": lang_scripts,
        "sources": sources,
    }

    final_mappings[lang] = mappings

with open("nllb_pairs.json", "w") as f:
    json.dump(nllb_mappings, f, indent=4)

# with open("language_mappings.json", "w") as f:
#     json.dump(final_mappings, f, indent=4)
