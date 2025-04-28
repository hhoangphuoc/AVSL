import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    Speech2TextTokenizer,
)
from avhubert.src.model.avhubert2text import AV2TextForConditionalGeneration
from avhubert.src.model.av2text_config import AV2TextConfig
from avhubert.src.dataset.load_data import load_feature


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AVHuBERT for ASR")
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        default="nguyenvulebinh/AV-HuBERT-MuAViC-en",
        help="Path to pretrained AVHuBERT model or identifier"
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True,
        default="checkpoints/hf-avhubert",
        help="Path to pretrained AVHuBERT model or identifier"
    )
    parser.add_argument(
        "--config_name", type=str, required=False,
        help="Path to AVHuBERTConfig or identifier"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        default="ami",
        help="HuggingFace dataset name "
    )
    parser.add_argument(
        "--dataset_config_name", type=str, default=None,
        help="HuggingFace dataset config variant"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        default="output/avhubert_ft",
        help="Directory to store fine-tuned model"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Learning rate for training"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model config and model
    config = AV2TextConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = AV2TextForConditionalGeneration.from_pretrained(
        args.model_name_or_path, config=config
    )

    # Load processor (feature extractor + tokenizer)
    tokenizer = Speech2TextTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    # Load dataset
    dataset_path = os.path.join("data", args.dataset_name, "dataset") #data/ami/dataset
    dataset = load_from_disk(dataset_path)

    print(dataset)


    # Preprocessing function
    def prepare_batch(batch):
        # Load audio and resample if needed
        audio = batch["audio"]
        audio_path = batch["audio"]["path"]
        
        input_values = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        with processor.as_target_processor():
            labels = processor(batch["text"]).input_ids
        batch["input_values"] = input_values
        batch["labels"] = labels
        return batch

    # Apply preprocessing
    dataset = dataset.map(
        prepare_batch,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

    # Data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True
    )

    # Compute metrics
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
        pred_str = processor.batch_decode(pred_ids)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        evaluation_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=100,
        save_steps=500,
        fp16=torch.cuda.is_available(),
        predict_with_generate=False,
        push_to_hub=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
