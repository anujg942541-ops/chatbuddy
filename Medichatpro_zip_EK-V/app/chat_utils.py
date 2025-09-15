from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load a small open model (good for Colab free tier)
# You can change this to a larger model like "mistralai/Mistral-7B-Instruct-v0.2"
# if you enable GPU in Colab.
MODEL_ID = "facebook/opt-350m"  

def get_chat_model(api_key=None):
    """
    Loads a Hugging Face model into a pipeline and wraps it for LangChain.
    api_key is ignored (kept for compatibility with your main.py).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    # Text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True
    )

    # Wrap in LangChain's HuggingFacePipeline
    return HuggingFacePipeline(pipeline=pipe)


def ask_chat_model(chat_model, prompt: str) -> str:
    """
    Sends a prompt to the Hugging Face model and returns the response as text.
    """
    result = chat_model.invoke(prompt)
    if isinstance(result, str):
        return result
    elif hasattr(result, "content"):
        return result.content
    else:
        return str(result)

# #Activation of LLM 
# from euriai.langchain import create_chat_model

# API_KEY = None
# MODEL = "gpt-4.1-nano"
# TEMPERATURE = 0.7

# def get_chat_model(api_key: str = None):
#     return create_chat_model(
#         api_key=api_key or API_KEY,
#         model=MODEL,
#         temperature=TEMPERATURE
#     )

# def ask_chat_model(chat_model, prompt: str):
#     response = chat_model.invoke(prompt)
#     return response.content 

