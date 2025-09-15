from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

# Pick your model here 👇
#MODEL_ID = "facebook/opt-350m"
#MODEL_ID = "tiiuae/falcon-7b-instruct"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"


def get_chat_model(api_key=None):
    """
    Loads a Hugging Face model into a pipeline and wraps it for LangChain.
    NOTE: these models need GPU for best performance (enable GPU in Colab).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",   # puts model on GPU if available
        torch_dtype="auto"   # uses fp16/bf16 automatically if supported
    )

    # Text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True
    )

    return HuggingFacePipeline(pipeline=pipe)


def extract_direct_answer(prompt: str, context: str) -> str | None:
    """
    Try to directly extract answers from the context for factual queries
    (like surgeries, age, medications). Returns None if no match.
    """
    if not context:
        return None
    text = context.lower()

    # Surgery extraction
    if "surgery" in prompt.lower() or "surgeries" in prompt.lower():
        match = re.search(r"surgeries?:\s*(.*)", context, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    # Age extraction
    if "age" in prompt.lower():
        match = re.search(r"age:\s*\d+", context, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    # Medications extraction
    if "medication" in prompt.lower() or "drug" in prompt.lower():
        match = re.search(r"medications?:\s*(.*)", context, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    # Allergies extraction
    if "allerg" in prompt.lower():
        match = re.search(r"allergies?:\s*(.*)", context, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return None


def ask_chat_model(chat_model, prompt: str, context: str = None) -> str:
    """
    Sends a prompt to the model, but first tries direct extraction from context
    to avoid hallucinations.
    """
    if context:
        extracted = extract_direct_answer(prompt, context)
        if extracted:
            return extracted

    # Fallback: let the LLM answer
    result = chat_model.invoke(prompt)
    if isinstance(result, str):
        return result
    elif hasattr(result, "content"):
        return result.content
    else:
        return str(result)
