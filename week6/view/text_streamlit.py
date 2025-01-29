import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
from models import get_model


# API KEY 정보로드
load_dotenv()
model = get_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("이미지를 업로드 해주세요"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        messages = []
        # 기존의 message들을 모두 포함하여 prompt 준비
        for m in st.session_state.messages:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))

        result = model.invoke(messages)
        response = result.content.split("<start_of_turn>")[-1]
        st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
