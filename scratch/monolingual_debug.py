from wqe import (
    MultilingualLoader,
    HfSentencePieceTokenizer,
    TokenizerConfig,
)

loader = MultilingualLoader(["hau", "yor", "pcm", "ibo", "swa"])
loader.load(load_path="WikiQuality/pre_filtered_top100", streaming=True)
loader.apply_language_sampling(sampling_strategy="temperature", temperature=3)

tokenizer_config = TokenizerConfig(model={"type": "unigram"}, vocab_size=10000)

tokenizer = HfSentencePieceTokenizer.train_from_config(
    loader["train"], tokenizer_config
)

pass
