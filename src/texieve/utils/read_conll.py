import conllu
from glob import glob
import datasets
from datasets import Dataset, Features
    

def _generate_examples(filepath):
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
                        "idx": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        "tokens": datasets.Sequence(datasets.Value("string")),
                        "lemmas": datasets.Sequence(datasets.Value("string")),
                        "upos": datasets.Sequence(
                            datasets.features.ClassLabel(
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
                        "xpos": datasets.Sequence(datasets.Value("string")),
                        "feats": datasets.Sequence(datasets.Value("string")),
                        "head": datasets.Sequence(datasets.Value("string")),
                        "deprel": datasets.Sequence(datasets.Value("string")),
                        "deps": datasets.Sequence(datasets.Value("string")),
                        "misc": datasets.Sequence(datasets.Value("string")),
                    })
        )
        
        dataset[k] = split
        
    return datasets.DatasetDict(dataset)



