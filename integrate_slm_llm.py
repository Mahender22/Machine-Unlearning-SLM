# integrate_slm_llm.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from config import BASE_LLM_MODEL_NAME, SLM_MODEL_DIR, DEVICE, HF_AUTH_TOKEN

class SLM_LLM_Integrator:
    def __init__(self, base_llm_model_name, hf_auth_token=None):
        # Load the SLM
        self.slm_tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_DIR)
        self.slm_model = AutoModelForSequenceClassification.from_pretrained(SLM_MODEL_DIR)
        self.slm_model.to(DEVICE)

        # Load the base LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(base_llm_model_name, use_auth_token=hf_auth_token)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            base_llm_model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            use_auth_token=hf_auth_token,
        )
        self.llm_model.to(DEVICE)

    def detect_forbidden_content(self, text):
        inputs = self.slm_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = self.slm_model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        return prediction == 1  # Returns True if forbidden content is detected

    def generate_response(self, user_input):
        if self.detect_forbidden_content(user_input):
            return "I'm sorry, but I don't have information on that topic."
        else:
            input_ids = self.llm_tokenizer.encode(user_input, return_tensors='pt').to(DEVICE)
            outputs = self.llm_model.generate(
                input_ids,
                max_length=256,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.llm_tokenizer.eos_token_id,
            )
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response