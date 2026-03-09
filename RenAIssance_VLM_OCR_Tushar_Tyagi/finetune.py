"""
Fine-tuning module for Vision-Language Models (VLMs) on OCR tasks.

This script provides an end-to-end modular pipeline to fine-tune models like
Qwen/Qwen2-VL-7B-Instruct or meta-llama/Llama-3.2-11B-Vision-Instruct
for handwritten OCR using LoRA and 4-bit quantization.
"""

import os
import json
import argparse
import random
from typing import Dict, List, Any, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    PreTrainedModel,
    ProcessorMixin
)
from trl import SFTTrainer


def load_prompts(prompts_dir: str) -> List[str]:
    """Loads all text prompts from a directory for prompt diversity during training.

    Args:
        prompts_dir (str): Directory containing .txt files with prompt templates.

    Returns:
        List[str]: List of prompts loaded from the directory. Returns a default prompt
            if the directory doesn't exist or is empty.
    """
    prompts = []
    
    if os.path.exists(prompts_dir):
        for filename in sorted(os.listdir(prompts_dir)):
            if filename.endswith('.txt'):
                filepath = os.path.join(prompts_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                        if prompt:
                            prompts.append(prompt)
                            print(f"Loaded prompt from {filename}")
                except Exception as e:
                    print(f"Warning: Could not read prompt file {filename}: {e}")
    
    # Fallback to default if no prompts loaded
    if not prompts:
        print("No prompts found in directory. Using default prompt.")
        prompts = ["<image> Transcribe this handwritten text."]
    
    return prompts


def prepare_dataset(
    annotation_file: str,
    base_image_dir: str = "",
    extraction_prompts: List[str] = None
) -> Dataset:
    """Reads image and corresponding OCR text pairs and formats them for SFTTrainer.

    Assumes the annotation file is a JSONL where each line has the format:
    {"image": "relative/path/to/image.jpg", "text": "Ground truth OCR text"}

    Args:
        annotation_file (str): Path to the JSONL annotation file.
        base_image_dir (str): Base directory for the images. Defaults to "".
        extraction_prompts (List[str]): List of prompt templates to randomly select from
            during training for improved generalization. If None, uses a default prompt.

    Returns:
        Dataset: A Hugging Face Dataset containing conversational formatted data
            with 'messages' and 'image_path' columns.
    """
    if extraction_prompts is None:
        extraction_prompts = ["<image> Transcribe this handwritten text."]
    
    formatted_data: List[Dict[str, Any]] = []

    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            image_path = os.path.join(base_image_dir, item["image"])
            ocr_text = item["text"]

            # Verify the image exists before adding to dataset
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue

            # Randomly select a prompt for this sample to improve model generalization
            extraction_prompt = random.choice(extraction_prompts)

            # Construct the conversational format compatible with SFTTrainer
            # Use image index (0) to reference the image in the images list
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": extraction_prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ocr_text}
                    ]
                }
            ]
            formatted_data.append({"messages": messages, "image_path": image_path})

    return Dataset.from_list(formatted_data)


def initialize_model_and_processor(
    model_id: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05
) -> Tuple[PreTrainedModel, ProcessorMixin]:
    """Initializes the VLM model with 4-bit quantization and configures LoRA.

    Args:
        model_id (str): Hugging Face model identifier (e.g., 'Qwen/Qwen2-VL-7B-Instruct').
        lora_r (int): Rank parameter for LoRA targeting attention modules.
        lora_alpha (int): Alpha scaling parameter for LoRA.
        lora_dropout (float): Dropout probability for LoRA layers.

    Returns:
        Tuple[PreTrainedModel, ProcessorMixin]: The configured PEFT wrapped model 
            and its associated processor.
    """
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Configure 4-bit Quantization to minimize VRAM usage while maintaining precision
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Try loading with AutoModelForImageTextToText first (for vision-language models)
    # If that fails, use AutoModelForCausalLM (for standard LMs)
    model = None
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
    except ValueError as e:
        print(f"AutoModelForImageTextToText failed, trying AutoModelForCausalLM instead...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
        except ValueError as e:
            print(f"AutoModelForCausalLM failed, trying AutoModel instead...")
            # Try with quantization first
            try:
                model = AutoModel.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    dtype=torch.bfloat16,
                )
            except (ValueError, AttributeError) as e:
                print(f"AutoModel with quantization failed, loading without quantization...")
                # Fallback: load without quantization for models that don't support it
                model = AutoModel.from_pretrained(
                    model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    dtype=torch.bfloat16,
                )
                print("⚠ Model loaded without 4-bit quantization (will use full precision)")

    # Prepare model for 4-bit/8-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA targeting the essential attention projections
    # For VLMs, try with CAUSAL_LM first, fall back to no task_type for incompatible models
    try:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        peft_model = get_peft_model(model, lora_config)
    except AttributeError as e:
        print(f"LoRA with CAUSAL_LM task_type failed, trying without task_type...")
        # For VLMs that don't have prepare_inputs_for_generation
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
            lora_dropout=lora_dropout,
            bias="none"
        )
        peft_model = get_peft_model(model, lora_config)
    
    peft_model.print_trainable_parameters()

    return peft_model, processor

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import torch

def run_training(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    train_dataset: Dataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
    learning_rate: float = 2e-4,
) -> None:
    """Configures and executes the training loop using SFTTrainer."""

    def preprocess_fn(examples):
        """
        Pre-process each example into model inputs.
        Loads images and applies the chat template + processor in one shot,
        so TRL never needs to reconcile messages ↔ images itself.
        """
        all_input_ids = []
        all_attention_mask = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_labels = []

        messages_list = examples["messages"]
        image_paths   = examples["image_path"]

        for msgs, img_path in zip(messages_list, image_paths):
            img_path = str(img_path).strip()
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Warning: skipping {img_path}: {e}")
                continue

            # Build the prompt string with the chat template
            text = processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False
            )

            # Let the processor handle both text and image together
            inputs = processor(
                text=text,
                images=[image],
                return_tensors="pt",
                padding=False,
            )

            input_ids = inputs["input_ids"].squeeze(0)           # (seq_len,)
            attention_mask = inputs["attention_mask"].squeeze(0) # (seq_len,)

            # Build labels: mask the user/system tokens, keep only the
            # assistant response tokens so we don't train on the prompt.
            labels = input_ids.clone()

            # Find where the assistant turn starts by locating the
            # assistant header in the token sequence.
            assistant_token_ids = processor.tokenizer.encode(
                "<|im_start|>assistant", add_special_tokens=False
            )
            seq = input_ids.tolist()
            start_idx = len(seq)  # default: mask everything (safety fallback)
            for k in range(len(seq) - len(assistant_token_ids)):
                if seq[k:k + len(assistant_token_ids)] == assistant_token_ids:
                    start_idx = k + len(assistant_token_ids)
                    break

            labels[:start_idx] = -100  # mask prompt tokens

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

            # Vision tensors — squeeze the batch dim added by the processor
            all_pixel_values.append(inputs["pixel_values"].squeeze(0))
            if "image_grid_thw" in inputs:
                all_image_grid_thw.append(inputs["image_grid_thw"].squeeze(0))

        return {
            "input_ids":       all_input_ids,
            "attention_mask":  all_attention_mask,
            "pixel_values":    all_pixel_values,
            "image_grid_thw":  all_image_grid_thw if all_image_grid_thw else [None] * len(all_input_ids),
            "labels":          all_labels,
        }

    print("Pre-processing dataset (tokenising + loading images)...")
    processed_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=8,
        remove_columns=train_dataset.column_names,
    )
    processed_dataset.set_format("torch")

    # ------------------------------------------------------------------ #
    #  Collator: pad variable-length sequences inside each mini-batch     #
    # ------------------------------------------------------------------ #
    def collate_fn(batch):
        input_ids      = [b["input_ids"]      for b in batch]
        attention_mask = [b["attention_mask"]  for b in batch]
        labels         = [b["labels"]          for b in batch]
        pixel_values   = [b["pixel_values"]    for b in batch]

        # Pad text tensors to the longest sequence in the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=processor.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        # Stack vision tensors (all images are the same shape after processor)
        pixel_values = torch.stack(pixel_values)

        result = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "pixel_values":   pixel_values,
        }

        # Include grid metadata if present (required by Qwen2.5-VL)
        if batch[0].get("image_grid_thw") is not None:
            result["image_grid_thw"] = torch.stack(
                [b["image_grid_thw"] for b in batch]
            )

        return result

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Use vanilla Trainer — no formatting_func, no SFTTrainer multimodal magic needed.
    # The dataset is already fully processed; Trainer just needs to collate + forward.
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=collate_fn,
    )

    print("Initiating fine-tuning phase...")
    trainer.train()

    print(f"Saving finalized LoRA adapter weights to: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


def main():
    """Main execution block to orchestrate the end-to-end OCR VLM fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Vision-Language Models for OCR tasks"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Hugging Face model ID to fine-tune (default: Qwen/Qwen2-VL-7B-Instruct)"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="data/train_annotations.jsonl",
        help="Path to JSONL annotation file with image and text pairs (default: data/train_annotations.jsonl)"
    )
    parser.add_argument(
        "--base_image_dir",
        type=str,
        default="data/",
        help="Base directory for image files (default: data/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/ocr_vlm_lora",
        help="Directory to save the fine-tuned LoRA adapter weights (default: ./checkpoints/ocr_vlm_lora)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device training batch size (default: 2)"
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for optimizer (default: 2e-4)"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank parameter (default: 16)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling parameter (default: 32)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability (default: 0.05)"
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="prompts",
        help="Directory containing .txt prompt files for diverse training (default: 'prompts')"
    )

    args = parser.parse_args()

    # Validate that annotation file exists
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file '{args.annotation_file}' not found.")
        print("Please provide a valid path using --annotation_file argument.")
        return

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Annotation File: {args.annotation_file}")
    print(f"  Base Image Directory: {args.base_image_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}\n")

    print("Step 1/3: Preparing the dataset ecosystem...")
    # Load prompts from the specified directory for improved generalization
    extraction_prompts = load_prompts(args.prompts_dir)
    print(f"Loaded {len(extraction_prompts)} prompt(s) for training diversity.\n")
    
    train_dataset = prepare_dataset(
        annotation_file=args.annotation_file,
        base_image_dir=args.base_image_dir,
        extraction_prompts=extraction_prompts
    )
    print(f"Successfully loaded dataset with {len(train_dataset)} instances.")

    print(f"Step 2/3: Bootstrapping VLM '{args.model}' and processor...")
    model, processor = initialize_model_and_processor(
        model_id=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("Step 3/3: Commencing the training loop...")
    run_training(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate
    )
    
    print("Fine-tuning pipeline execution concluded successfully.")


if __name__ == "__main__":
    main()
