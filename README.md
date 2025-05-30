Fine Tunine with peft--LoRA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A demonstration of Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA) on transformer-based language models. This project provides a minimal example of fine-tuning a pre-trained model on the Shakespeare dataset using Hugging Face’s `transformers`, `datasets`, and `peft` libraries.

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
* [Example Notebook](#example-notebook)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Features

* **Parameter-Efficient**: Fine-tune only a small subset of model parameters via LoRA.
* **Low GPU Memory Footprint**: Reduce GPU memory usage, enabling fine-tuning on modest hardware.
* **Reproducible Example**: End-to-end notebook showcasing data loading, model preparation, training, and evaluation.

## Prerequisites

* Python 3.8+
* CUDA-enabled GPU (optional, but recommended for faster training)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Rishi-Kora/peft-fine-tuning-LoRA.git
   cd peft-fine-tuning-LoRA
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install transformers datasets peft torch accelerate
   ```

## Usage

To fine-tune a model with LoRA, follow these steps:

1. Prepare the dataset:

   ```python
   from datasets import load_dataset
   dataset = load_dataset("tiny_shakespeare")
   ```

2. Load a pre-trained model and tokenizer:

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_name = "gpt2"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   ```

3. Apply LoRA configuration:

   ```python
   from peft import LoraConfig, get_peft_model

   lora_config = LoraConfig(
       r=16,
       lora_alpha=32,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05,
       bias="none",
       task_type="CAUSAL_LM",
   )
   peft_model = get_peft_model(model, lora_config)
   ```

4. Fine-tune with Hugging Face Trainer:

   ```python
   from transformers import Trainer, TrainingArguments

   training_args = TrainingArguments(
       output_dir="./lora-output",
       num_train_epochs=3,
       per_device_train_batch_size=8,
       save_steps=500,
       logging_steps=100,
       fp16=True,
   )

   trainer = Trainer(
       model=peft_model,
       args=training_args,
       train_dataset=dataset["train"],
   )
   trainer.train()
   ```

## Example Notebook

See the Jupyter notebook [`fine_tune_shakespeare.ipynb`](fine_tune_shakespeare.ipynb) for a detailed, step-by-step walkthrough of the fine-tuning process.

## Results

After training, you can generate text with the fine-tuned model:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="./lora-output", tokenizer=tokenizer)
print(generator("ROMEO:\nO Juliet, wherefore art thou", max_length=100))
```

Sample output:

```
ROMEO:
O Juliet, wherefore art thou sweet love so fair,
That in the very light thy beauty glows...
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes, enhancements, or additional examples.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, contact **Rishi Kora** at **[rishikora@example.com](mailto:korarishi@gmail.com)**.
