import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import BASE_LLM_MODEL_NAME, DATA_DIR, HF_AUTH_TOKEN

def generate_examples(prompt, num_samples, model, tokenizer):
    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
    )

    # Move inputs to the model's device
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
            model.resize_token_embeddings(len(tokenizer))

    # Generate outputs
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=50,
        num_return_sequences=num_samples,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    examples = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_text = text[len(prompt):].strip()
        examples.append(generated_text)
    return examples

def generate_training_data(model_name, num_positive=500, num_negative=500, hf_auth_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        token=hf_auth_token,
    )

    # Generate positive examples
    positive_prompt = "Generate diverse questions that people might ask about Peter Parker."
    positive_examples = generate_examples(positive_prompt, num_positive, model, tokenizer)

    # Generate negative examples
    negative_prompt = "Generate diverse general knowledge questions unrelated to Peter Parker."
    negative_examples = generate_examples(negative_prompt, num_negative, model, tokenizer)

    # Save data to files
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'positive_examples.txt'), 'w', encoding='utf-8') as f:
        for example in positive_examples:
            f.write(example + '\n')

    with open(os.path.join(DATA_DIR, 'negative_examples.txt'), 'w', encoding='utf-8') as f:
        for example in negative_examples:
            f.write(example + '\n')

if __name__ == '__main__':
    if not BASE_LLM_MODEL_NAME:
        raise ValueError("BASE_LLM_MODEL_NAME is not set in config.py.")
    generate_training_data(BASE_LLM_MODEL_NAME, hf_auth_token=HF_AUTH_TOKEN)