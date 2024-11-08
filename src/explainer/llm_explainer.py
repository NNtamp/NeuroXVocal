from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch
from typing import Dict, List

class LLMExplainer:
    def __init__(self, model_name: str = "google/flan-t5-xl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device.type == 'cuda':
            try:
                # Optimized loading for better memory usage
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=True
                )

                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print("Model loaded with 8-bit quantization and FP16.")
            except Exception as e:
                print(f"Failed to load model with optimizations: {e}")
                print("Falling back to base loading.")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    device_map="auto"
                )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map=None
            )
            
        self.model.eval()

    def generate_explanation(self, prompt: str) -> str:
        """Generate explanation using the LLM with optimized parameters."""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)

            outputs = self.model.generate(
                inputs.input_ids,
                max_length=768,
                min_length=150,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                do_sample=True,
                no_repeat_ngram_size=3,
                num_beams=3,
                early_stopping=True
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return "An error occurred during explanation generation. Please try again."

    def __call__(self, prompt: str) -> str:
        """Convenience method to generate explanation."""
        return self.generate_explanation(prompt)