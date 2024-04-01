import os
import time

import streamlit as st
from agent import build_agent
from ingest import load_document

# Custom image for the app icon and the assistant's avatar
company_logo = "https://www.app.nl/wp-content/uploads/2019/01/Blendle.png"

# Configure Streamlit page
st.set_page_config(page_title="Your Document Explainer Chatbot", page_icon=company_logo)

document_path = None
uploaded_file = st.file_uploader("Upload Document", type=["pdf"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    document_path = "temp.pdf"


# Initialize chat history
if "messages" not in st.session_state:
    # Start with first message from assistant
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi human! I am your Doc Explainer AI. How can I help you today?",
        }
    ]
# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if document_path:
    processed_document = load_document(document_path)

    if query := st.chat_input("Ask me anything"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        agent = build_agent(processed_document)
        with st.chat_message("assistant", avatar=company_logo):
            message_placeholder = st.empty()
            # Send user's question to our agent
            response = agent.invoke(
                {"input": query},
                return_only_outputs=True,
            )
            response_output = response["output"]
            full_response = ""

            # Simulate stream of response with milliseconds delay
            for chunk in response_output.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant message to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_output,
            }
        )

    # Remove temporary file after processing
    if os.path.exists(document_path):
        os.remove(document_path)
