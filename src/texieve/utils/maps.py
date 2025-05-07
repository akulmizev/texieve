from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)

from ..data import metrics


METRIC_MAP = {
    "length_chars": metrics.LengthCharacters,
    "length_words": metrics.LengthWords,
    "unique_words": metrics.UniqueWords,
    "unique_trigrams": metrics.UniqueTrigrams,
    "unique_chars": metrics.UniqueCharacters,
    "alpha_chars": metrics.AlphaChars,
    "num_lines": metrics.NumLines,
    "frac_all_caps_words": metrics.FracAllCapsWords,
    "frac_no_script_words": metrics.FracNoScriptWords,
    "doc_mean_word_length": metrics.MeanWordLength,
    "frac_unique_words": metrics.FracUniqueWords,
    "frac_unique_trigrams": metrics.FracUniqueTrigrams,
    "frac_unique_chars": metrics.FracUniqueCharacters,
    "frac_lines_end_ellipsis": metrics.FracLinesEndEllipsis,
    "frac_symbol_to_words": metrics.FracSymbolToWords,
    "unigram_entropy": metrics.UnigramEntropy,
    "trigram_entropy": metrics.TrigramEntropy,
    "char_entropy": metrics.CharacterEntropy,
    "lines_end_with_punctuation": metrics.LinesEndWithPunctuation,
    "num_words_per_line": metrics.NumWordsPerLine,
    "perplexity": metrics.Perplexity,
}

TOKENIZER_PARAM_MAP = {
    "model": {
        "unigram": models.Unigram,
        "bpe": models.BPE,
        "wordpiece": models.WordPiece,
    },
    "normalizer": {
        "nfc": normalizers.NFC,
        "nfd": normalizers.NFD,
        "nfkc": normalizers.NFKC,
        "nfkd": normalizers.NFKD,
        "nmt": normalizers.Nmt,
        "prepend": normalizers.Prepend,
        "replace": normalizers.Replace,
        "strip": normalizers.Strip,
        "bert": normalizers.BertNormalizer,
    },
    "pre_tokenizer": {
        "byte_level": pre_tokenizers.ByteLevel,
        "metaspace": pre_tokenizers.Metaspace,
        "whitespace": pre_tokenizers.Whitespace,
        "whitespace_split": pre_tokenizers.WhitespaceSplit,
        "unicode_scripts": pre_tokenizers.UnicodeScripts,
        "punctuation": pre_tokenizers.Punctuation,
        "digits": pre_tokenizers.Digits,
        "bert": pre_tokenizers.BertPreTokenizer,
        "split": pre_tokenizers.Split,
    },
    "decoder": {
        "metaspace": decoders.Metaspace,
        "bpe": decoders.BPEDecoder,
        "wordpiece": decoders.WordPiece,
        "byte_level": decoders.ByteLevel,
    },
    "trainer": {
        "unigram": trainers.UnigramTrainer,
        "bpe": trainers.BpeTrainer,
        "wordpiece": trainers.WordPieceTrainer,
    },
}

TASK_TO_MODEL_AND_COLLATOR_MAPPING = {
    "mlm": {"model": AutoModelForMaskedLM, "collator": DataCollatorForLanguageModeling},
    "tagger": {
        "model": AutoModelForTokenClassification,
        "collator": DataCollatorForTokenClassification,
    },
    "classifier": {
        "model": AutoModelForSequenceClassification,
        "collator": DataCollatorWithPadding,
    },
}
