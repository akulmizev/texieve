import logging
import regex as re
import unicodedata
from glob import glob

from functools import wraps
from typing import Dict, Iterator, List, Pattern, Union

import conllu
from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    ClassLabel,
    IterableDataset,
    DatasetDict,
    IterableDatasetDict,
    concatenate_datasets,
)
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def compute_ngrams(words: List[Union[str, bytes]], n: int = 3) -> Iterator[str]:
    for gram in zip(*[words[i:] for i in range(n)]):
        yield " ".join(gram)


def tokenize(text: str) -> List[str]:
    # return re.findall(WHITE_SPACE, text, re.UNICODE)
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def batch_tokenize(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast = None,
    batch_size: int = 1000,
) -> Dataset:
    if tokenizer is None:
        tokenize_fn = tokenize
    else:
        tokenize_fn = tokenizer.tokenize

    tokenized_dataset = dataset.map(
        lambda x: {"tokens": tokenize_fn(x["text"])},
        batched=True,
        batch_size=batch_size,
    )

    return tokenized_dataset


def c4_filter(line: str, patterns: Dict[str, Pattern]) -> bool:
    """
    Filter for Common Crawl C4 dataset.
    """

    if not line.strip():
        return False

    if re.search(patterns["terminal_punct"], line[-1]) is None:
        return False

    if len(re.findall(patterns["tokens"], line)) < 5:
        return False

    if "lorem ipsum" in line.lower():
        return False

    return True


def get_all_punctuation():
    punctuation = []
    for codepoint in range(0x110000):  # Unicode range
        char = chr(codepoint)
        category = unicodedata.category(char)
        if category.startswith("P"):
            punctuation.append(char)
    return "".join(punctuation)


def measure_deletion(func):
    @wraps(func)
    def wrapper(self, dataset, **kwargs):
        is_iterable = isinstance(dataset, IterableDataset)

        if not is_iterable:
            initial_docs = len(dataset)
            full_text = "".join(dataset["text"])
            initial_chars = len(full_text)
            initial_bytes = len(full_text.encode("utf-8"))

        result_dataset = func(self, dataset, **kwargs)

        if not is_iterable:
            final_docs = len(result_dataset)
            full_text = "".join(result_dataset["text"])
            final_chars = len(full_text)
            final_bytes = len(full_text.encode("utf-8"))

            logger.info(
                f"Deleted: {initial_docs - final_docs} docs "
                f"({(initial_docs - final_docs) / initial_docs:.2%}), "
                f"{initial_chars - final_chars} chars "
                f"({(initial_chars - final_chars) / initial_chars:.2%}), "
                f"{(initial_bytes - final_bytes) / (1024 * 1024):.2f} MB "
                f"({(initial_bytes - final_bytes) / initial_bytes:.2%})"
            )

        return result_dataset

    return wrapper


def convert_iterable_dataset_to_regular(dataset: IterableDataset) -> Dataset:
    """
    Converts an IterableDataset to a Dataset.

    Parameters
    ----------
    dataset : IterableDataset
        The IterableDataset to convert.

    Returns
    -------
    Dataset
        The converted Dataset.
    """
    return Dataset.from_generator(
        lambda: (yield from dataset), features=dataset.features
    )


def add_column(
    dataset: Union[Dataset, IterableDataset], column_name: str, column_value: str
):
    """
    Adds a column to the dataset. Assumes all examples have the same value.

    Parameters
    ----------
    dataset : Union[Dataset, IterableDataset]
        The dataset to which the column will be added.
    column_name : str
        The name of the column to add.
    column_value : str
        The value to be added to all examples.

    Returns
    -------
    Union[Dataset, IterableDataset]
        The dataset with the added language column.
    """
    if isinstance(dataset, Dataset):
        return dataset.add_column(column_name, [column_value] * dataset.num_rows)
    elif isinstance(dataset, IterableDataset):
        return dataset.map(lambda example: {**example, column_name: column_value})
    else:
        raise ValueError(
            "Dataset backend must be a huggingface Dataset or IterableDataset"
        )


def get_column_names(
    dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
) -> List[str]:
    """
    Get column names from various dataset types.

    Parameters
    ----------
    dataset : Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]
        The dataset to get column names from.

    Returns
    -------
    List[str]
        A list of column names.

    Raises
    ------
    ValueError
        If the dataset type is not supported.
    """
    if isinstance(dataset, (Dataset, DatasetDict)):
        return (
            dataset.column_names
            if isinstance(dataset, Dataset)
            else dataset["train"].column_names
        )
    elif isinstance(dataset, (IterableDataset, IterableDatasetDict)):
        # For IterableDatasets, we need to peek at the first example
        if isinstance(dataset, IterableDataset):
            first_example = next(iter(dataset))
        else:
            first_example = next(iter(dataset["train"]))
        return list(first_example.keys())
    else:
        raise ValueError(
            "Unsupported dataset type. Please provide a Dataset, IterableDataset, or their Dict variants."
        )


def combine_datasets(
    datasets: Dict[str, List[Union[Dataset, IterableDataset]]], streaming: bool = False
) -> Union[DatasetDict, IterableDatasetDict]:
    """
    Combines multiple datasets into a single DatasetDict or IterableDatasetDict.

    Parameters
    ----------
    datasets : Dict[str, List[Union[Dataset, IterableDataset]]]
        A dictionary where keys are split names and values are lists of datasets to combine.
    streaming : bool, optional

    Returns
    -------
    Union[DatasetDict, IterableDatasetDict]
        The combined dataset.
    """
    data = {
        split: concatenate_datasets(datasets_list)
        for split, datasets_list in datasets.items()
    }

    if streaming:
        return IterableDatasetDict(data)
    else:
        return DatasetDict(data)
    
    
def _generate_examples(filepath):
    """Generator function to yield examples from a CoNLL-U file for Hugging Face datasets."""
    id = 0
    with open(filepath, "r", encoding="utf-8") as data_file:
        tokenlist = list(conllu.parse_incr(data_file))
        for sent in tokenlist:
            if "sent_id" in sent.metadata:
                idx = sent.metadata["sent_id"]
            else:
                idx = id

            tokens = [token["form"] for token in sent]

            if "text" in sent.metadata:
                txt = sent.metadata["text"]
            else:
                txt = " ".join(tokens)

            yield {
                "idx": str(idx),
                "text": txt,
                "tokens": [token["form"] for token in sent],
                "lemmas": [token["lemma"] for token in sent],
                "upos": [token["upos"] for token in sent],
                "xpos": [token["xpos"] for token in sent],
                "feats": [str(token["feats"]) for token in sent],
                "head": [str(token["head"]) for token in sent],
                "deprel": [str(token["deprel"]) for token in sent],
                "deps": [str(token["deps"]) for token in sent],
                "misc": [str(token["misc"]) for token in sent],
            }
            id += 1
            
def conllu_to_datasets(UD_DIR: str):
    """Convert CoNLL-U files in a directory to a Hugging Face DatasetDict."""
    
    train_file = glob(f"{UD_DIR}/*train.conllu")[0]
    dev_file = glob(f"{UD_DIR}/*dev.conllu")[0]
    test_file = glob(f"{UD_DIR}/*test.conllu")[0]
    
    dataset = {"train": train_file, "validation": dev_file, "test": test_file}
    for k, v in dataset.items():
        split = Dataset.from_generator(
            _generate_examples,
            gen_kwargs={"filepath": v},
            features=Features(
                    {
                        "idx": Value("string"),
                        "text": Value("string"),
                        "tokens": Sequence(Value("string")),
                        "lemmas": Sequence(Value("string")),
                        "upos": Sequence(ClassLabel(
                                names=[
                                    "NOUN",
                                    "PUNCT",
                                    "ADP",
                                    "NUM",
                                    "SYM",
                                    "SCONJ",
                                    "ADJ",
                                    "PART",
                                    "DET",
                                    "CCONJ",
                                    "PROPN",
                                    "PRON",
                                    "X",
                                    "_",
                                    "ADV",
                                    "INTJ",
                                    "VERB",
                                    "AUX",
                                ]
                            )
                        ),
                        "xpos": Sequence(Value("string")),
                        "feats": Sequence(Value("string")),
                        "head": Sequence(Value("string")),
                        "deprel": Sequence(Value("string")),
                        "deps": Sequence(Value("string")),
                        "misc": Sequence(Value("string")),
                    })
        )
        
        dataset[k] = split
        
    return DatasetDict(dataset)




