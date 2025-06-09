import streamlit as st
import requests
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

def main():

    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="📑",
        layout="centered"
    )
    st.title("📝 Chat With Maritime Standards")

    with st.sidebar:
        st.title("Chatbot Info")
        st.markdown("#### Base Model")
        st.markdown("GPT-4o")
        st.markdown("#### Uploaded Documents")
        st.markdown("- [ABS: Rules for building and classing, Marine vessels, Part 4](https://ww2.eagle.org/content/dam/eagle/rules-and-guides/current/other/1-rules-for-building-and-classing-marine-vessels-2025/1-mvr-part-4-jan25.pdf)")
        st.markdown("- [BV: Rules for the classification of steel ships, Part C](hhttps://erules.veristar.com/dy/data/bv/pdf/467-NR_PartC_2025-01.pdf)")
        st.markdown("- DNV: Rules for classification, Part 4")
        st.markdown("#### RAG parameters")
        n_chunks = st.slider("Select number of retrieved chunks", min_value=1, max_value=10, value=4, step=1)
        collection = st.selectbox("Select the database collection", ["All", "ABS", "BV", "DNV"])

    # Initialize chat history in session state if not already defined.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages using Streamlit's chat message containers.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "image_base64" in message:
                image_data = base64.b64decode(message["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                st.image(image, caption=message.get("image_caption", ""), use_container_width=True)
            
            st.markdown(message["content"])

    # Chat input widget for the user to type their question.
    prompt = st.chat_input("Enter your question here")
    if prompt:
        # Append the user's message to session state.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Set your backend URL. Adjust if your backend is hosted elsewhere.
        backend_url = os.getenv("BACKEND_URL")
        payload = {"question": prompt, "n_chunks": n_chunks, "collection": collection}
        
        try:
            response = requests.post(f"{backend_url}/message", json=payload)
            response.raise_for_status()
            data = response.json()
            ai_response = data.get("ai_response", "No response received.")
        except Exception as e:
            ai_response = f"Error: {e}"
            data = {}
        
        assistant_message = {"role": "assistant", "content": ai_response}

        if "image_base64" in data:
            assistant_message["image_base64"] = data["image_base64"]
            assistant_message["image_caption"] = data.get("image_caption", "")

        # Append the assistant's response to session state.
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            if "image_base64" in data:
    
                image_data = base64.b64decode(data["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                st.image(image, caption=data["image_caption"], use_container_width=True)

            st.markdown(ai_response)


if __name__ == "__main__":
    main()
