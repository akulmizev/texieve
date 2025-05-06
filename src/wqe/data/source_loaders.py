import json
from typing import List
import importlib.resources as pkg_resources
from datasets import load_dataset, Dataset, concatenate_datasets, interleave_datasets

from . import resources


def load_wiki(lang_wiki_id: str, dump_date: str, streaming: bool) -> Dataset:
    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"{dump_date}.{lang_wiki_id}",
        streaming=streaming,
    )
    return dataset.remove_columns(["title", "id"])


def load_c4(lang_bcp_47: str, streaming: bool) -> Dataset:
    dataset = load_dataset("allenai/c4", lang_bcp_47, streaming=streaming)
    return dataset.remove_columns(["timestamp"])


def load_bible(lang_id: str, streaming: bool) -> Dataset:
    dataset = load_dataset(
        "davidstap/biblenlp-corpus-mmteb",
        f"eng-{lang_id}",
        trust_remote_code=True,
        streaming=streaming,
    )
    dataset = dataset.remove_columns(["eng"])
    return dataset.rename_column(lang_id, "text")


def get_text(example, tag):
    example["text"] = example["translation"][tag]
    return example


def load_nllb(lang_id: str, scripts: List[str], streaming: bool) -> Dataset:
    with pkg_resources.open_text(resources, "nllb_pairs.json") as file:
        nllb_pairs = json.load(file)
    bcp_tags = {f"{lang_id}_{script}" for script in scripts}
    datasets = []
    for trans_pair in nllb_pairs[lang_id]:
        matching_tag = next(
            (tag for tag in trans_pair.split("-") if tag in bcp_tags), None
        )
        if matching_tag:
            dataset = load_dataset(
                "allenai/nllb",
                trans_pair,
                trust_remote_code=True,
                streaming=streaming,
            )["train"]
            dataset = dataset.map(get_text, fn_kwargs={"tag": matching_tag})
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col != "text"]
            )
            datasets.append(dataset)

    if streaming:
        return interleave_datasets(datasets, stopping_strategy="first_exhausted")
    else:
        return concatenate_datasets(datasets)

def load_fineweb(lang_id: str, streaming: bool) -> Dataset:
    with pkg_resources.open_text(resources, "fineweb_mappings.json") as file:
        fineweb_subsets = json.load(file)
    
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        fineweb_subsets[lang_id],
        trust_remote_code=True,
        streaming=streaming,
    )
    for split in dataset.keys():
        dataset[split] = dataset[split].remove_columns(
            [col for col in dataset[split].column_names if col != "text"]
        )

    return dataset