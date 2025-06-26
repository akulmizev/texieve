import logging

from typing import Union, Dict

import evaluate
import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
    AutoConfig,
    set_seed,
)

from .base import ModelFromConfig
from ..tokenization import HfTokenizerFromConfig
from ..utils.config import TrainingParameters
from ..utils.biaffine import (
    BertForBiaffineParsing,
    RobertaForBiaffineParsing,
    DebertaForBiaffineParsing,
    UD_HEAD_LABELS
)

__all__ = ["Tagger", "Classifier", "BiaffinePraser"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tagger(ModelFromConfig):
    """
    Class for token-level classification tasks, such as
    Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
    Only NER and POS are supported for now.

    Parameters
    ----------
    load_path : str
        Path to load the model from (either a local path or a Hugging Face Hub path).
    config : TrainingParameters
        Configuration object for training parameters.
    **kwargs
        Additional keyword arguments for the parent class.

    Attributes
    ----------
    label_set : List[str]
        List of labels for the task.
        Assumes the `tags` feature in the dataset.
    metrics : evaluate.EvaluationModule
        Evaluation metrics for the task.

    Methods
    -------
    _init_model_and_tokenizer(dataset=None, tokenization=None)
        Initializes the model and tokenization for the task.
    _init_metrics()
        Initializes the evaluation metrics for the task.
    _align_labels(example)
        Aligns the token-level labels with the tokenized input.
    _tokenize_and_collate(dataset)
        Tokenizes and collates a dataset into a PyTorch DataLoader.
    _eval_loop(loader)
        Performs an evaluation loop on the given DataLoader and returns the evaluation scores.
    _eval_loop_ner(loader)
        Performs an evaluation loop for the NER task.
    _eval_loop_pos(loader)
        Performs an evaluation loop for the POS tagging task.
    """

    def __init__(self, load_path: str, config: TrainingParameters, **kwargs):
        super().__init__(load_path, config, **kwargs)

        self._init_metrics()

    def _init_model_and_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizerFast, HfTokenizerFromConfig] = None,
        label_set: set = None,
    ):
        """
        Initialize the model and tokenization for tagging.

        Parameters
        ----------
        tokenizer : Union[PreTrainedTokenizerFast, HfTokenizerFromConfig], optional
            The tokenization to be used. Generally not needed, as the tokenization will be loaded
            from the same path as the model.
        label_set : set, optional
            Set of labels for the task.
        """

        self.label_set = label_set

        if self.seed:
            set_seed(seed=self.seed)

        self._model = AutoModelForTokenClassification.from_pretrained(
            self.load_path,
            num_labels=len(self.label_set),
            id2label={i: label for i, label in enumerate(self.label_set)},
            label2id={label: i for i, label in enumerate(self.label_set)},
        )

        self.tokenizer = (
            tokenizer
            if tokenizer
            else PreTrainedTokenizerFast.from_pretrained(self.load_path)
        )
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )

        logger.info(f"{self._model.config.model_type} for {self.task} loaded.")
        logger.info(
            f"Number of parameters: {round(self._model.num_parameters() / 1e6)}M"
        )

    def _init_metrics(self):
        """
        Initialize the evaluation metrics for the task.
        """

        if self.task == "ner":
            self.metrics = evaluate.load("seqeval")
        elif self.task == "pos":
            # Not adding f1 because, at time of writing, the zero_division argument
            # is not supported in `evaluate` for f1, despite being implemented
            # in the underlying `sklearn` function. It will be computed manually in
            # the `_eval_loop_pos` method.
            self.metrics = evaluate.combine(["precision", "recall"])
        else:
            raise ValueError(
                f"Task {self.task} not supported. Only 'ner' and 'pos' are supported for now."
            )

    def _align_labels(self, example):
        """
        Align the token-level labels with the tokenized input.
        Have to ignore special tokens and common prefixes/suffixes.

        Parameters
        ----------
        example : dict
            A single example from the dataset.

        Returns
        -------
        dict
            The tokenized input with aligned labels.
        """

        TO_IGNORE = ["Ġ", "▁", "##", "Ċ"] + list(
            self.tokenizer.special_tokens_map.values()
        )

        tokenized_input = self.tokenizer(
            example["tokens"],
            padding=self.padding_strategy,
            max_length=self.max_length,
            truncation=True,
            is_split_into_words=True,
        )

        seen = set()
        labels = []
        for i, word_id in enumerate(tokenized_input.word_ids()):
            token_id = tokenized_input["input_ids"][i]
            tag_id = example["tags"][word_id] if word_id is not None else -100
            if self.tokenizer.convert_ids_to_tokens(token_id) in TO_IGNORE:
                labels.append(-100)
            elif word_id in seen:
                labels.append(-100)
            else:
                labels.append(tag_id)
                seen.add(word_id)

        tokenized_input["labels"] = labels

        return tokenized_input

    def _tokenize_and_collate(self, dataset):
        """
        Tokenize and collate a dataset into a PyTorch DataLoader.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be tokenized and collated.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader containing the tokenized and collated dataset.
        """

        batched_dataset = dataset.map(
            self._align_labels, remove_columns=dataset.column_names
        )

        loader = DataLoader(
            batched_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            pin_memory=True,
        )

        return loader

    def _eval_loop(self, loader):
        """
        Perform an evaluation loop on the given DataLoader and return scores.
        Have to differentiate between NER and POS tagging tasks since `seqeval` only supports NER.

        Parameters
        ----------
        loader : DataLoader
            A PyTorch DataLoader containing the data to be evaluated.

        Returns
        -------
        dict
            A dictionary containing the evaluation scores (precision, recall, f1).
        """

        if self.task == "ner":
            return self._eval_loop_ner(loader)
        elif self.task == "pos":
            return self._eval_loop_pos(loader)

    def _eval_loop_ner(self, loader):
        """
        Perform an evaluation loop for NER.

        Parameters
        ----------
        loader : DataLoader
            A PyTorch DataLoader containing the data to be evaluated.

        Returns
        -------
        dict
            A dictionary containing the evaluation scores (precision, recall, f1) for NER.
        """

        self._model.eval()

        for batch in loader:
            with torch.no_grad():
                outputs = self._model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            idx_to_keep = batch["labels"] != -100

            filtered_preds = [
                list(
                    map(
                        self.label_set.__getitem__,
                        preds[i][idx_to_keep[i]].detach().cpu().tolist(),
                    )
                )
                for i in range(len(preds))
            ]

            filtered_labels = [
                list(
                    map(
                        self.label_set.__getitem__,
                        batch["labels"][i][idx_to_keep[i]].detach().cpu().tolist(),
                    )
                )
                for i in range(len(preds))
            ]

            self.metrics.add_batch(
                predictions=filtered_preds, references=filtered_labels
            )

        scores = self.metrics.compute(zero_division=0.0)

        return {
            "precision": scores["overall_precision"],
            "recall": scores["overall_recall"],
            "f1": scores["overall_f1"],
        }

    def _eval_loop_pos(self, loader):
        """
        Perform an evaluation loop for POS.

        Parameters
        ----------
        loader : DataLoader
            A PyTorch DataLoader containing the data to be evaluated.

        Returns
        -------
        dict
            A dictionary containing the evaluation scores (precision, recall, f1) for POS.
        """

        self._model.eval()

        for batch in loader:
            with torch.no_grad():
                outputs = self._model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            idx_to_keep = batch["labels"] != -100

            filtered_preds = preds[idx_to_keep].detach().cpu().tolist()
            filtered_labels = batch["labels"][idx_to_keep].detach().cpu().tolist()

            self.metrics.add_batch(
                predictions=filtered_preds, references=filtered_labels
            )

        scores = self.metrics.compute(average="weighted", zero_division=0.0)

        return {
            "precision": scores["precision"],
            "recall": scores["recall"],
            # See note in `_init_metrics` method for why this is computed manually
            "f1": scores["precision"]
            * scores["recall"]
            / (scores["precision"] + scores["recall"])
            * 2,
        }


class Classifier(ModelFromConfig):
    """
    Class for sequence classification tasks.

    Parameters
    ----------
    load_path : str
        Path to load the model from (either a local path or a Hugging Face Hub path).
    config : TrainingParameters
        Configuration object for training parameters.
    **kwargs
        Additional keyword arguments for the parent class.

    Attributes
    ----------
    label_set : List[str]
        List of labels for the task.
        Assumes the `labels` feature in the dataset.
    metrics : evaluate.EvaluationModule
        Evaluation metrics for the task.

    Methods
    -------
    _init_model_and_tokenizer(dataset=None, tokenization=None)
        Initializes the model and tokenization for the task.
    _init_metrics()
        Initializes the evaluation metrics for the task.
    _tokenize_and_collate(dataset)
        Tokenizes and collates a dataset into a PyTorch DataLoader.
    _eval_loop(loader)
        Performs an evaluation loop on the given DataLoader and returns the evaluation scores.
    """

    def __init__(self, load_path: str, config: TrainingParameters, **kwargs):
        super().__init__(load_path, config, **kwargs)

        self._init_metrics()

    def _init_model_and_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizerFast, HfTokenizerFromConfig] = None,
        label_set: set = None,
    ):
        """
        Initialize the model and tokenization for classification.

        Parameters
        ----------
        tokenizer : Union[PreTrainedTokenizerFast, HfTokenizerFromConfig], optional
            The tokenization to be used. Generally not needed, as the tokenization will be loaded
            from the same path as the model.
        label_set : set, optional
            Set of labels for the task.
        """

        self.label_set = label_set

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.load_path,
            num_labels=len(self.label_set),
            id2label={i: label for i, label in enumerate(self.label_set)},
            label2id={label: i for i, label in enumerate(self.label_set)},
        )

        self.tokenizer = (
            tokenizer
            if tokenizer
            else PreTrainedTokenizerFast.from_pretrained(self.load_path)
        )
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )

        logger.info(f"{self._model.config.model_type} for {self.task} loaded.")
        logger.info(
            f"Number of parameters: {round(self._model.num_parameters() / 1e6)}M"
        )

    def _init_metrics(self):
        """
        Initialize the evaluation metrics for the task.
        """

        self.metrics = evaluate.combine(["precision", "recall"])

    def _tokenize_and_collate(self, dataset):
        """
        Tokenize and collate a dataset into a PyTorch DataLoader.
        Assumes `text` and `labels` are features in the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be tokenized and collated.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader containing the tokenized and collated dataset.
        """
        if "premise" in dataset.features:
            batched_dataset = dataset.map(
                lambda examples: self.tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    padding=self.padding_strategy,
                    max_length=self.max_length,
                    truncation=True,
                ),
                batched=True,
                remove_columns=[
                    column for column in dataset.column_names if column != "labels"
                ],
            )
        else:
            batched_dataset = dataset.map(
                lambda examples: self.tokenizer(
                    examples["text"],
                    padding=self.padding_strategy,
                    max_length=self.max_length,
                    truncation=True,
                ),
                batched=True,
                remove_columns=[
                    column for column in dataset.column_names if column != "labels"
                ],
            )

        batched_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        loader = DataLoader(
            batched_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            pin_memory=True,
        )

        return loader

    def _eval_loop(self, loader):
        """
        Perform an evaluation loop on the DataLoader and return scores (precision, recall, f1).

        Parameters
        ----------
        loader : DataLoader
            A PyTorch DataLoader containing the data to be evaluated.

        Returns
        -------
        dict
            A dictionary containing the evaluation scores (precision, recall, f1).
        """

        self._model.eval()

        for batch in loader:
            with torch.no_grad():
                outputs = self._model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            self.metrics.add_batch(predictions=preds, references=labels)

        scores = self.metrics.compute(average="weighted", zero_division=0.0)

        return {
            "precision": scores["precision"],
            "recall": scores["recall"],
            # See note in `_init_metrics` method for why this is computed manually
            "f1": scores["precision"]
            * scores["recall"]
            / (scores["precision"] + scores["recall"])
            * 2,
        }
        
class BiaffineParser(ModelFromConfig):
    """
    Class for dependency parsing using biaffine attention mechanism.
    Parameters
    ----------
    load_path : str
        Path to load the model from (either a local path or a Hugging Face Hub path).
    config : TrainingParameters
        Configuration object for training parameters.
    **kwargs
        Additional keyword arguments for the parent class.
        
    Attributes
    ----------
    label_set : List[str]
        List of labels for the task.
    
    Methods
    -------
    _init_model_and_tokenizer(tokenizer=None, label_set=None)
        Initializes the model and tokenization for dependency parsing.
    _align_labels(example)
        Aligns the arc_labels and rel_labels labels with the tokenized input.
    _tokenize_and_collate(dataset)
        Tokenizes and collates a dataset into a PyTorch DataLoader.
    _compute_metrics(arc_preds, rel_preds, arc_labels, rel_labels)
        Computes the evaluation metrics (las and uas) for the dependency parsing task.
    _eval_loop(loader)
        Performs an evaluation loop on the given DataLoader and returns the evaluation scores.
    """
    def __init__(self, load_path: str, config: TrainingParameters, **kwargs):
        super().__init__(load_path, config, **kwargs)

        self.model_type = config.model_type
                
    def _init_model_and_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizerFast, HfTokenizerFromConfig] = None,
        label_set: set = set(UD_HEAD_LABELS),
    ):
        """
        Initialize the model and tokenization for biaffine parsing.
        Parameters
        ----------
        tokenizer : Union[PreTrainedTokenizerFast, HfTokenizerFromConfig], optional
            The tokenization to be used. Generally not needed, as the tokenization will be loaded
            from the same path as the model.
        label_set : set, optional
            Set of labels for the task.
        """
        
        # TODO check if it is okay to include all labels or this needs to be data specifc
        self.label_set = list(label_set)
        self.id2label: Dict[int, str] = {i: label for i, label in enumerate(self.label_set)}
        self.label2id: Dict[str, int] = {label: i for i, label in enumerate(self.label_set)}
        
        dp_config = AutoConfig.from_pretrained(
            self.load_path,
            num_labels=len(self.label_set),
            id2label=self.id2label,
            label2id=self.label2id,
            # TODO: make these configurable
            attention_probs_dropout_prob=0.1, 
            hidden_dropout_prob=0.1
        )
        
        if self.model_type == "bert":
            self._model = BertForBiaffineParsing.from_pretrained(
                self.load_path,
                config=dp_config
            )
        elif self.model_type == "roberta":
            self._model = RobertaForBiaffineParsing.from_pretrained(
                self.load_path,
                config=dp_config
            )
        elif self.model_type == "deberta":
            self._model = DebertaForBiaffineParsing.from_pretrained(
                self.load_path,
                config=dp_config
            )
        else:
            raise ValueError(
                f"Model type {self.model_type} not supported. "
                "Only 'bert', 'roberta', and 'deberta' are supported for now."
            )
        
        self.tokenizer = (
            tokenizer
            if tokenizer
            else PreTrainedTokenizerFast.from_pretrained(self.load_path)
        )
        
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )

        logger.info(f"{self._model.config.model_type} for {self.task} loaded.")
        logger.info(
            f"Number of parameters: {round(self._model.num_parameters() / 1e6)}M"
        )
    
    def _align_labels(self, example):
        """ 
        Aligns the arc_labels and rel_labels labels with the tokenized input.
        Have to ignore special tokens and common prefixes/suffixes.
        
        Parameters
        ----------
        example : dict
            A single example from the dataset.
            
        Returns
        -------
        dict
            The tokenized input with aligned arc_labels, rel_labels, and word_starts.
        """
        
        TO_IGNORE = ["Ġ", "▁", "##", "Ċ"] + list(
            self.tokenizer.special_tokens_map.values()
        )

        tokenized_input = self.tokenizer(
            example["tokens"],
            padding=self.padding_strategy,
            max_length=self.max_length,
            truncation=True,
            is_split_into_words=True
        )

        
        seen = set()
        arc_labels = []
        rel_labels = []
        word_starts = []
                
        previous_word_id = None
        for i, word_id in enumerate(tokenized_input.word_ids()):
            if word_id is not None and word_id != previous_word_id:
                word_starts.append(i)  
            previous_word_id = word_id
            token_id = tokenized_input["input_ids"][i]
            arc_label = int(example["head"][word_id]) if word_id is not None else -100
            rel_label = self.label2id[example["deprel"][word_id]] if word_id is not None else -100
            if self.tokenizer.convert_ids_to_tokens(token_id) in TO_IGNORE:
                arc_labels.append(-100)
                rel_labels.append(-100)
            elif word_id in seen:
                arc_labels.append(-100)
                rel_labels.append(-100)
            else:
                arc_labels.append(arc_label)
                rel_labels.append(rel_label)
                seen.add(word_id)
                
        tokenized_input["arc_labels"] = arc_labels
        tokenized_input["rel_labels"] = rel_labels
        tokenized_input["word_starts"] = word_starts + (self.max_length + 1 - len(word_starts)) * [-100]
        
                     
        return tokenized_input
    
    def _tokenize_and_collate(self, dataset):
        """ 
        Tokenizes and collates a dataset into a PyTorch DataLoader.
        Assumes `tokens`, `head`, and `deprel` are features in the dataset. 
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to be tokenized and collated.
        
        Returns
        -------
        DataLoader
            A PyTorch DataLoader containing the tokenized and collated dataset.
        """
        
        batched_dataset = dataset.map(
            self._align_labels, remove_columns=dataset.column_names
        )

        loader = DataLoader(
            batched_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            pin_memory=True,
        )

        return loader
    
    def _compute_metrics(self, arc_preds, rel_preds, arc_labels, rel_labels):
        """
        Computes the evaluation metrics (las and uas) for the dependency parsing task.
        
        Parameters
        ----------
        arc_preds : List[int]
            List of predicted arc labels.
        rel_preds : List[int]
            List of predicted relation labels.
        arc_labels : List[int]
            List of true arc labels.
        rel_labels : List[int]
            List of true relation labels.
            
        Returns
        -------
        dict
            A dictionary containing the evaluation scores (uas and las).
        """
        correct_arcs = np.equal(arc_preds, arc_labels)
        correct_rels = np.equal(rel_preds, rel_labels)
        correct_arcs_and_rels = correct_arcs * correct_rels
        
        unlabeled_correct = correct_arcs.sum()
        labeled_correct = correct_arcs_and_rels.sum()
        total_words = correct_arcs.size
        
        unlabeled_attachment_score = unlabeled_correct / total_words
        labeled_attachment_score = labeled_correct / total_words
        
        return {
            "uas": unlabeled_attachment_score * 100,
            "las": labeled_attachment_score * 100,
        }
    
    def _eval_loop(self, loader):
        """
        Performs an evaluation loop on the given DataLoader and returns the evaluation scores.
        
        Parameters
        ----------
        loader : DataLoader
            A PyTorch DataLoader containing the data to be evaluated.   
        
        Returns
        -------
        dict
            A dictionary containing the evaluation scores (uas and las).
        """
        
        self._model.eval()
        
        arc_preds, rel_preds = [], []
        arc_labels, rel_labels = [], []
        
        for batch in loader:
            with torch.no_grad():
                outputs = self._model(**batch)
                
            mask = batch["arc_labels"].ne(-100)
            
            _arc_labels = batch["arc_labels"][mask]
            _rel_labels = batch["rel_labels"][mask]
            
            _arc_preds = torch.argmax(outputs.arc_logits, dim=-1)[mask]
            _rel_preds = outputs.rel_logits[mask]
            _rel_preds = _rel_preds[torch.arange(len(_arc_labels)), _arc_labels]
            _rel_preds = torch.argmax(_rel_preds, dim=-1)
            

            arc_preds.extend(_arc_preds.detach().cpu().tolist())
            rel_preds.extend(_rel_preds.detach().cpu().tolist())
            arc_labels.extend(_arc_labels.detach().cpu().tolist())
            rel_labels.extend(_rel_labels.detach().cpu().tolist())
            
        
        scores = self._compute_metrics(arc_preds, rel_preds, arc_labels, rel_labels)
        
        return scores
