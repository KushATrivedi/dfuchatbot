import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertModel
import time


# Load the fine-tuned model
model_path = "kusht55/dfu_chatbot"
#token = os.getenv("HUGGING_FACE_TOKEN")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.push_to_hub("kusht55/dfu_chatbot")
BertModel.from_pretrained(model_path, use_auth_token=True)


# Set page title and favicon
st.set_page_config(page_title="Diabetic Foot Ulcer Chatbot", page_icon=":hospital:")

# Set app title and description
st.title("Diabetic Foot Ulcer Chatbot")
st.markdown("Welcome to the Diabetic Foot Ulcer Chatbot. Ask any questions related to diabetic foot ulcers!")

# Define conversation loop
def chatbot(user_input):
    # Check for exit command
    if user_input.lower() == 'exit':
        st.info("Chat ended. Goodbye!")
        return

    # Display typing animation
    with st.spinner(text="Chatbot is typing..."):
        time.sleep(2)  # Simulate typing time

        # Get model response
        response = generate_response(user_input)

    # Display model response
    st.text_area("Chatbot:", value=response, height=100, max_chars=500)

# Generate response from the model
def generate_response(user_input):
    # Encode user input
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate model response
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )

    # Decode and return model response
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Get user input
user_input = st.text_input("You:")

# Start conversation when user submits input
if st.button("Send"):
    chatbot(user_input)
