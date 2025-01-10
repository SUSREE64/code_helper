import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
st.title("Python Code Helper")

try:
    info = st.empty()
    info.markdown("#### :red[Model is Loading....]")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    info.markdown("#### :green[Model Loaded Successfully]")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input and form handling
st.markdown("### Python Code Generation")
with st.form(key="code_form"):
    prompt = st.text_area("Enter your coding prompt:", height=200)
    submit = st.form_submit_button("Generate Code")

    if submit and prompt.strip():
        with st.spinner("Generating response..."):
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.markdown("### Generated Code:")
                st.code(response, language="python")
            except Exception as e:
                st.error(f"An error occurred: {e}")
