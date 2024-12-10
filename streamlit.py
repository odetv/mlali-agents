import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import runModel


EXAMPLE_QUESTIONS = [
    "Saya ingin liburan, tapi belum tau harus kemana",
    "Berikan saya rekomendasi wisata di ubud",
    "Saya ingin ke pura besakih, apa yang perlu disiapkan?",
    "Saya dari SGR ke KINTAMANI ingin mencari glamping"
]
INITIAL_MESSAGE = {"role": "assistant", "content": "Ada yang bisa saya bantu?", "raw_content": "Ada yang bisa saya bantu?"}


def setup_page():
    st.set_page_config(page_title="Mlali Agents")
    st.title("Yuk Mlali😊")


def process_response(prompt):
    with st.spinner("Sedang proses..."):
        _, response = runModel(prompt)
        return response


def add_message(role, content, html_content=None):
    if "messages" not in st.session_state:
        st.session_state.messages = [INITIAL_MESSAGE]
    message = {
        "role": role,
        "content": html_content if html_content else content,
        "raw_content": content
    }
    st.session_state.messages.append(message)


def display_example_questions():
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, prompt in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(prompt):
                add_message("user", prompt)
                st.session_state['user_question'] = prompt
                response = process_response(prompt)
                add_message("assistant", response)
                st.rerun()


def display_chat_history():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["raw_content"])


def handle_user_input():
    if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
        add_message("user", prompt)
        st.session_state['user_question'] = prompt
        st.chat_message("user").write(prompt)
        response = process_response(prompt)
        add_message("assistant", response)
        st.chat_message("assistant").markdown(response)


def main():
    setup_page()
    display_example_questions()
    st.markdown("***")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [INITIAL_MESSAGE]
    
    display_chat_history()
    handle_user_input()

if __name__ == "__main__":
    main()