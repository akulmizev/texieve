### Fine-tune a trained model on a downstream task:

```python
from datasets import load_dataset
from texieve import HfTokenizerFromConfig, TrainingParameters, Classifier

# Load dataset
masakhanews = load_dataset("masakhanews", "hau")
masakhanews = mashakanews.rename_columns("label", "labels")

# If using a locally-saved dataset, you can load it like this:
# 
# from datasets import load_from_disk
# masakhanews = load_from_disk("your-path/masakhanews")
#
# Note, however, that custom dataset configurations (e.g. language subsets) 
# must be specified directly via, e.g.: "your-path/masakhanews/hau"

# Load tokenization
tokenizer = HfTokenizerFromConfig.from_pretrained("./models/unigram_tokenizer")

# Specify training parameters
params = TrainingParameters(
    model_type="deberta",
    task="classification",
    max_length=512,
    num_train_epochs=10,
    batch_size=32,
    lr=1e-3,
    padding_strategy="max_length"
)

# Initialize the model (this can also be a huggingface model identifier, such as "microsoft/deberta-base")
deberta_classifier = Classifier(load_path="./models/deberta_mlm", config=params)

# Train the model
deberta_classifier.train(masakhanews, tokenizer, eval_split="validation")

# Test the model
deberta_classifier.test(masakhanews, split="test")
```

Currently supported tasks are:

- `classification`: text classification with `Classifier`
- `nli`: natural language inference with `Classifier`
- `pos`: part-of-speech tagging with `Tagger`
- `ner`: named entity recognition with `Tagger`
- `ud` : dependency parsing with `BiaffineParser`

Note that both `Classifier` and `Tagger` expect the label fields to be `labels` and `tags`, respectively. 
In the case that a dataset does not use these fields, you can rename them using the `rename_columns` 
method from the `datasets` library. 

Since the finetuning pipeline is designed to work with the `datasets` library, we have provided a script to convet UD datasets from `.conllu` files to a `DatasetDict` in `src/utils/read_conll.py`. Example usage:
```python
# import the function 
from utils.read_conll import conllu_to_datasets

# this returns a DatasetDict object with train, validation and test splits that can be directly passed to BiaffineParser.train()
ud_dataset = conllu_to_datasets(<path-to-ud-folder>) 
```