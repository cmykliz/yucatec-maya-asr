"""
finetune_mms.py
---------------
Fine-tunes facebook/mms-300m on the DoReCo Yucatec Maya dataset
using CTC training via HuggingFace Trainer.

Usage:
    python finetune_mms.py \
        --data_dir ./doreco_yuca1254_hf \
        --output_dir ./mms-300m-yucatec-maya \
        --num_train_epochs 30 \
        --per_device_train_batch_size 8

After training, evaluate with:
    python finetune_mms.py --eval_only --output_dir ./mms-300m-yucatec-maya

Requirements:
    pip install transformers datasets evaluate jiwer accelerate
"""

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
import evaluate

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal) GPU")
        return "mps"
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return "cuda"
    else:
        print("Using CPU")
        return "cpu"


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

# DoReCo uses angle-bracket markers for non-speech events — remove them
NOISE_PATTERN = re.compile(r"<<[^>]*>>|<[^>]*>")

def clean_transcription(text: str) -> str:
    """Remove DoReCo noise markers and normalise whitespace."""
    text = NOISE_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()  # lowercase for CTC consistency


# ─────────────────────────────────────────────
# VOCABULARY BUILDING
# ─────────────────────────────────────────────

def build_vocabulary(dataset, output_dir: str) -> str:
    """
    Extract all unique characters from training transcriptions
    and write vocab.json for the CTC tokenizer.

    Returns path to vocab.json.
    """
    vocab_set = set()
    for example in dataset["train"]:
        text = clean_transcription(example["transcription"])
        vocab_set.update(text)

    # Remove empty string if present
    vocab_set.discard("")

    # Build vocab dict — reserve indices for special tokens
    vocab_dict = {char: idx for idx, char in enumerate(sorted(vocab_set))}

    # Add required special tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    vocab_dict["|"]     = len(vocab_dict)  # word boundary token

    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary: {len(vocab_dict)} tokens → {vocab_path}")
    print(f"Sample chars: {sorted(list(vocab_set))[:20]}")
    return vocab_path


# ─────────────────────────────────────────────
# DATASET PREPARATION
# ─────────────────────────────────────────────

def prepare_dataset(batch, processor):
    """
    Process a batch:
      - Resample audio to 16kHz
      - Extract input values via feature extractor
      - Tokenize transcription → labels
    """
    audio = batch["audio"]

    # Extract input features from raw audio
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np"
    ).input_values[0]

    # Tokenize transcription
    text = clean_transcription(batch["transcription"])
    # Replace spaces with word boundary token
    text = text.replace(" ", "|")
    batch["labels"] = processor.tokenizer(text).input_ids

    return batch


# ─────────────────────────────────────────────
# DATA COLLATOR
# ─────────────────────────────────────────────

@dataclass
class DataCollatorCTCWithPadding:
    """
    Pads input_values and labels to the longest sequence in the batch.
    Labels are padded with -100 so they are ignored in the CTC loss.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": f["input_values"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding token id with -100 (ignored by CTC loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch


# ─────────────────────────────────────────────
# WER METRIC
# ─────────────────────────────────────────────

def make_compute_metrics(processor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids    = np.argmax(pred_logits, axis=-1)

        # Replace -100 in labels (padding) with pad token id
        pred.label_ids[pred.label_ids == -100] = \
            processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(
            pred.label_ids, group_tokens=False
        )

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(args):
    device = get_device()

    # ── 1. Load dataset ───────────────────────────────────────────────
    print("\nLoading dataset...")
    ds = load_dataset("audiofolder", data_dir=args.data_dir)
    # Cast audio to 16kHz (MMS requirement)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(ds)

    # ── 2. Build vocabulary ───────────────────────────────────────────
    print("\nBuilding vocabulary...")
    vocab_path = build_vocabulary(ds, args.output_dir)

    # ── 3. Build processor (tokenizer + feature extractor) ────────────
    print("\nBuilding processor...")
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(args.output_dir)

    # ── 4. Preprocess dataset ─────────────────────────────────────────
    print("\nPreprocessing dataset (this may take a few minutes)...")
    ds = ds.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=ds["train"].column_names,
        num_proc=1,  # keep at 1 for MPS compatibility
    )

    # ── 5. Load pretrained MMS model ──────────────────────────────────
    print(f"\nLoading {args.model_name}...")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,  # needed when replacing vocab head
    )

    # Freeze the feature encoder — only fine-tune the transformer layers
    # This is recommended for small datasets to prevent overfitting
    model.freeze_feature_encoder()
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── 6. Training arguments ─────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,       # lower WER is better
        fp16=False,                    # MPS doesn't support fp16
        dataloader_num_workers=0,      # required for MPS stability
        report_to="none",              # disable wandb
        push_to_hub=False,
    )

    # ── 7. Data collator ──────────────────────────────────────────────
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True
    )

    # ── 8. Trainer ────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
    )

    if args.eval_only:
        # ── Evaluation only ───────────────────────────────────────────
        print("\nRunning evaluation only...")
        metrics = trainer.evaluate()
        print(f"\nTest WER: {metrics['eval_wer']:.4f}")
    else:
        # ── Baseline WER (zero-shot, before fine-tuning) ──────────────
        print("\nComputing baseline WER (before fine-tuning)...")
        baseline = trainer.evaluate()
        print(f"Baseline WER: {baseline['eval_wer']:.4f}")

        # ── Training ──────────────────────────────────────────────────
        print("\nStarting fine-tuning...")
        trainer.train()

        # ── Final evaluation ──────────────────────────────────────────
        print("\nFinal evaluation...")
        metrics = trainer.evaluate()
        print(f"\n{'='*50}")
        print(f"Baseline WER : {baseline['eval_wer']:.4f}")
        print(f"Fine-tuned WER: {metrics['eval_wer']:.4f}")
        improvement = (baseline['eval_wer'] - metrics['eval_wer']) / baseline['eval_wer'] * 100
        print(f"Improvement  : {improvement:.1f}%")
        print(f"{'='*50}")

        # Save final model
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"\nModel saved to {args.output_dir}")

        # Save results summary
        results = {
            "model": args.model_name,
            "language": "yua",
            "language_name": "Yucatec Maya",
            "dataset": "DoReCo 2.0",
            "train_hours": 0.89,
            "test_hours": 0.22,
            "baseline_wer": baseline["eval_wer"],
            "finetuned_wer": metrics["eval_wer"],
            "improvement_pct": round(improvement, 1),
            "epochs": args.num_train_epochs,
        }
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune MMS for Yucatec Maya ASR"
    )
    parser.add_argument("--data_dir",    default="./doreco_yuca1254_hf",
                        help="Path to HuggingFace audiofolder dataset")
    parser.add_argument("--output_dir",  default="./mms-300m-yucatec-maya",
                        help="Where to save model checkpoints and results")
    parser.add_argument("--model_name",  default="facebook/mms-300m",
                        help="Pretrained model (facebook/mms-300m or facebook/mms-1b)")
    parser.add_argument("--num_train_epochs",          type=int,   default=30)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--eval_only",   action="store_true",
                        help="Skip training, just evaluate a saved model")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                    help="Learning rate (default 1e-5)")
    args = parser.parse_args()
    main(args)
