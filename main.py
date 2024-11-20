# main.py

from config import BASE_LLM_MODEL_NAME, HF_AUTH_TOKEN
from integrate_slm_llm import SLM_LLM_Integrator

def main():
    # User selects the base LLM
    print("Select the base LLM model:")
    print("1. Llama 2 7B")
    print("2. Llama 2 13B")
    print("3. Custom model")
    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        BASE_LLM_MODEL_NAME = 'meta-llama/Llama-2-7b-hf'
    elif choice == '2':
        BASE_LLM_MODEL_NAME = 'meta-llama/Llama-2-13b-hf'
    elif choice == '3':
        BASE_LLM_MODEL_NAME = input("Enter the model name or path: ")
    else:
        print("Invalid choice.")
        return

    # Update the config with the selected model
    import config
    config.BASE_LLM_MODEL_NAME = BASE_LLM_MODEL_NAME

    # Check if the model requires an authentication token
    requires_auth = False
    if 'meta-llama' in BASE_LLM_MODEL_NAME:
        requires_auth = True

    if requires_auth:
        hf_auth_token = input("Enter your Hugging Face authentication token: ")
        config.HF_AUTH_TOKEN = hf_auth_token
    else:
        config.HF_AUTH_TOKEN = None

    # Initialize the integrator
    integrator = SLM_LLM_Integrator(BASE_LLM_MODEL_NAME, hf_auth_token=config.HF_AUTH_TOKEN)

    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        response = integrator.generate_response(user_input)
        print(f"Assistant: {response}")

if __name__ == '__main__':
    main()