from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch
from typing import Dict, List

"""
This module handles the interaction with the LLM model and generates
explanations for patient diagnoses. Uses FLAN-T5, a fully open-source model
that doesn't require authentication.
"""


class LLMExplainer:
    def __init__(self, model_name: str = "google/flan-t5-xl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device.type == 'cuda':
            try:

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                print("Model loaded with 8-bit quantization using bitsandbytes.")
            except Exception as e:
                print(f"Failed to load model with bitsandbytes on GPU: {e}")
                print("Falling back to loading the model without quantization.")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    device_map="auto"
                )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map=None,
            )
            print("Model loaded without quantization for CPU.")

        self.model.to(self.device)

    def generate_explanation(self, prompt: str) -> str:
        """Generate explanation using the LLM."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            inputs.input_ids,
            max_length=512,
            min_length=100,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            do_sample=True,
            no_repeat_ngram_size=3
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
