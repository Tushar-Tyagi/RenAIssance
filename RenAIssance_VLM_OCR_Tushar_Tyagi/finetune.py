"""
Fine-tuning module for Vision-Language Models (VLMs) on OCR tasks.

This script provides an end-to-end modular pipeline to fine-tune models like
Qwen/Qwen2-VL-7B-Instruct or meta-llama/Llama-3.2-11B-Vision-Instruct
for handwritten OCR using LoRA and 4-bit quantization.
"""

import os
import json
from typing import Dict, List, Any, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    PreTrainedModel,
    ProcessorMixin
)
from trl import SFTTrainer


def prepare_dataset(
    annotation_file: str,
    base_image_dir: str = "",
    extraction_prompt: str = "<image> Transcribe this handwritten text."
) -> Dataset:
    """Reads image and corresponding OCR text pairs and formats them for SFTTrainer.

    Assumes the annotation file is a JSONL where each line has the format:
    {"image": "relative/path/to/image.jpg", "text": "Ground truth OCR text"}

    Args:
        annotation_file (str): Path to the JSONL annotation file.
        base_image_dir (str): Base directory for the images. Defaults to "".
        extraction_prompt (str): The prompt provided to the user role to extract text.

    Returns:
        Dataset: A Hugging Face Dataset containing conversational formatted data
            with a 'messages' column.
    """
    formatted_data: List[Dict[str, List[Dict[str, Any]]]] = []

    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            image_path = os.path.join(base_image_dir, item["image"])
            ocr_text = item["text"]

            # Construct the conversational format compatible with SFTTrainer
            # The structure relies on the multimodal chat format supported by modern processors.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
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
            formatted_data.append({"messages": messages})

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

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare model for 4-bit/8-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA targeting the essential attention projections
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model, processor


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
    """Configures and executes the training loop using SFTTrainer.

    Args:
        model (PreTrainedModel): The PEFT-wrapped VLM to fine-tune.
        processor (ProcessorMixin): The model's corresponding processor.
        train_dataset (Dataset): The dataset formatted for conversational SFT.
        output_dir (str): Directory where the trained LoRA adapter weights will be saved.
        epochs (int): Number of training epochs to run.
        batch_size (int): Per-device training batch size.
        grad_accum_steps (int): Gradient accumulation steps to maintain stability on consumer GPUs.
        learning_rate (float): Learning rate for the optimizer.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True, # bfloat16 maximizes stability and performance on modern GPU hardware
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False, # Must be False for multimodal models handling images
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
        args=training_args,
        dataset_text_field="messages",
        max_seq_length=1024,
    )

    print("Initiating fine-tuning phase...")
    trainer.train()

    print(f"Saving finalized LoRA adapter weights to: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


def main():
    """Main execution block to orchestrate the end-to-end OCR VLM fine-tuning."""
    # Example relative paths and configurations
    annotation_file = "data/train_annotations.jsonl"
    base_image_dir = "data/images/"
    output_dir = "./checkpoints/ocr_vlm_lora"
    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    if not os.path.exists(annotation_file):
        print(f"Warning: Annotation file '{annotation_file}' not found.")
        print("Please ensure your dataset is prepared or adjust the paths in main().")
        print("This script provides the structural framework for fine-tuning.")
        return

    print("Step 1/3: Preparing the dataset ecosystem...")
    train_dataset = prepare_dataset(
        annotation_file=annotation_file,
        base_image_dir=base_image_dir
    )
    print(f"Successfully loaded dataset with {len(train_dataset)} instances.")

    print(f"Step 2/3: Bootstrapping VLM '{model_id}' and processor...")
    model, processor = initialize_model_and_processor(model_id=model_id)

    print("Step 3/3: Commencing the training loop...")
    run_training(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        output_dir=output_dir
    )
    
    print("Fine-tuning pipeline execution concluded successfully.")


if __name__ == "__main__":
    main()
