# Multilingual BERT (mBERT) Training

This example demonstrates how to train a multilingual BERT model using the `texieve` package. 
The training process involves loading Wikipedia data for multiple languages, configuring the tokenizer, 
and training a masked language model (MLM) with the parameters specified in the BERT paper.

This script can be called via `accelerate` for distributed training like so:
```bash
accelerate launch --num_processes=4 --mixed_precision=bf16 train_mbert.py
```

```python
# train_mbert.py
from texieve import (
    MultilingualLoader, 
    HfTokenizerFromConfig,
    TokenizerConfig,
    MLM, 
    TrainingParameters
)

# Load Wikipedia data for multiple languages
language_codes = [
    'eng', 'ceb', 'deu', 'swe', 'fra', 'nld', 'rus', 'spa', 'ita', 'arz',
    'pol', 'jpn', 'zho', 'ukr', 'vie', 'war', 'ara', 'por', 'fas', 'cat',
    'srp', 'ind', 'kor', 'nor', 'che', 'fin', 'tur', 'ces', 'hun', 'tat',
    'srp', 'ron', 'eus', 'msa', 'epo', 'heb', 'hye', 'dan', 'bul', 'cym',
    'uzb', 'azb', 'slk', 'est', 'kaz', 'bel', 'min', 'ell', 'lit', 'hrv',
    'urd', 'glg', 'aze', 'slv', 'lld', 'kat', 'nno', 'hin', 'tam', 'tha',
    'ben', 'mkd', 'lat', 'yue', 'ast', 'lav', 'afr', 'tgk', 'mya', 'sqi',
    'mlg', 'mar', 'bos', 'oci', 'tel', 'mal', 'bel', 'bre', 'nds', 'kir',
    'swa', 'lmo', 'jav', 'new', 'pnb', 'hat', 'vec', 'pms', 'bak', 'kmr',
    'ltz', 'sun', 'gle', 'isl', 'szl', 'fry', 'ckb', 'chv', 'pan'
]

data = MultilingualLoader(language_codes)
# When working with many languages, we recommend to use locally-saved datasets 
# and load them in streaming mode
data.load(load_path='experiments/mbert_filtered_1', streaming=True)
data.apply_language_sampling(sampling_strategy='temperature', temperature=1.43)
data.save('../experiments/mbert')

# Specify BERT tokenizer parameters
tokenizer_config = TokenizerConfig(
    model={'type': 'wordpiece'},
    trainer={'type': 'wordpiece'},
    normalizer={'type': 'bert'},
    pre_tokenizer={'type': 'bert'},
    decoder={'type': 'bert'},
    vocab_size=110000,
    special_tokens={
        'cls_token': '[CLS]',
        'sep_token': '[SEP]',
        'unk_token': '[UNK]',
        'pad_token': '[PAD]',
        'mask_token': '[MASK]'
    }
)

# Train the tokenizer on Wiki data
tokenizer = HfTokenizerFromConfig.train_from_config(data, tokenizer_config)
tokenizer.save_pretrained('../experiments/mbert')

# Specify training parameters
model_load_path = '../config/model/mbert/config.json'
checkpoint_path = '../experiments/mbert'
model_config = TrainingParameters(
    model_type='bert',
    task='mlm',
    max_length=512,
    mask_prob=0.12,
    num_train_steps=800000,
    num_eval_steps=25000,
    num_train_epochs=10,
    batch_size=64,
    lr=1e-04,
    padding_strategy='longest',
    num_warmup_steps=10000,
    mixed_precision='bf16'
)

# Initialize the model from scratch
model = MLM(
    load_path=model_load_path,
    config=model_config,
    checkpoint_path=checkpoint_path
)

# Optionally initalize a wandb logging session
model.init_wandb(
    project='mbert',
    entity='WikiQuality',
    config=model_config,
)

# Train the model
model.train(data, tokenizer=tokenizer, eval_split='test')

# Save the model
model.save(checkpoint_path)
```
