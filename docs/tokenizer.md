### Train a tokenizer on wiki data:

```python
from texieve import MonolingualLoader, TokenizerConfig, HfTokenizerFromConfig

# Load Wikipedia data
wiki = MonolingualLoader("hau").load_dataset(sources=["wiki"], streaming=True)

# Create a tokenization configuration
tokenizer_config = TokenizerConfig(
    model={"type": "unigram"},
    trainer={"type": "unigram"},
    normalizer={"type": "nkfc"},
    pre_tokenizer=[
        {"type": "whitespace"},
        {"type": "digits"},
        {"type": "metaspace", "prepend_scheme": "always", "replacement": "▁"}
    ],
    decoder={"type": "metaspace"},
    vocab_size=10000
)

# Train tokenization
tokenizer = HfTokenizerFromConfig.train_from_config(wiki, tokenizer_config)

# Save tokenization
tokenizer.save_pretrained("./models/unigram_tokenizer")
```
This will train a (fast) UnigramLM tokenizer with full compatibility with
the popular `PreTrainedTokenizerFast` class from `transformers`. This process makes use of the
`tokenizers` [api](https://huggingface.co/docs/tokenizers/en/index), with support for custom normalizers, 
pre-tokenizers, decoders, and trainers. 
Here, all possible arguments for each component (e.g. pre-tokenizer) are passed directly onto the
associated `tokenizers` object during initialization. Where supported, `Sequence` objects can be passed to the
config as a list of multiple components (e.g., the `pre_tokenizer` argument above.) 

_Note:_ it is possible to set the `vocab_size` to `auto` to automatically determine the
optimal vocabulary size based on the data (via Heap's Law and additional heuristics).

### `sentencepiece` backend

It is also possible to train a huggingface-compatible tokenizer using a `sentencepiece` backend, like so:

```python
from texieve import HfSentencepieceTokenizer

tokenizer_config = TokenizerConfig(
    model={"type": "unigram"}, 
    vocab_size="auto"
)

tokenizer = HfSentencepieceTokenizer.train_from_config(wiki, tokenizer_config)
```

Instead of specifying components directly, tokenization hyperpameters, 
custom `sentencepiece` arguments (see [here](https://github.com/google/sentencepiece/blob/master/doc/options.md))
can be passed to the constructor. As above, the trainer tokenizer fully supports the `PreTrainedTokenizerFast` 
class from `transformers`, and can be saved using the `save_pretrained` method.
