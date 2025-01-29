import streamlit as st
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# OpenAI API 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 모델 초기화
chat_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# GPU 지원 여부 확인 (Apple Silicon 최적화)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# CLIP 모델 로드 (세션 유지하여 메모리 절약)
if "clip_model" not in st.session_state:
    st.session_state.clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)
    st.session_state.clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

clip_model = st.session_state.clip_model
clip_processor = st.session_state.clip_processor

# FAISS 벡터 저장소 초기화 (한 번만 실행)
if "vector_db" not in st.session_state:
    faiss_index = faiss.IndexFlatL2(512)  # 512차원 벡터 저장
    st.session_state.vector_db = faiss_index
    st.session_state.image_metadata = {}  # FAISS에 저장된 이미지 정보 (파일명 저장)

# 채팅 메시지 저장소 초기화 (채팅 기록 유지)
if "messages" not in st.session_state:
    st.session_state.messages = []


# CLIP을 사용하여 이미지 벡터화
def get_image_embedding(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten()


# 이미지를 Base64 인코딩하는 함수
def encode_image_to_base64(image_data):
    return base64.b64encode(image_data).decode("utf-8")


# FAISS에서 중복 이미지 확인
def find_duplicate_image(image_vector):
    """FAISS에서 중복된 이미지가 있는지 확인하고, 있으면 해당 인덱스를 반환"""
    if st.session_state.vector_db.ntotal == 0:
        return None  # 저장된 벡터가 없으면 중복 아님

    image_vector = np.array(image_vector, dtype=np.float32).reshape(1, -1)
    D, I = st.session_state.vector_db.search(image_vector, 1)  # 가장 가까운 벡터 검색

    if D[0][0] < 0.001:  # 거리가 0에 가까우면 동일한 이미지로 판단
        return I[0][0]  # 중복된 이미지의 인덱스 반환
    return None


# 이미지 업로드
uploaded_files = st.file_uploader(
    "이미지를 업로드 해주세요 (여러 개 가능)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        # CLIP 벡터 생성
        image_vector = get_image_embedding(image)
        image_vector = np.array(image_vector, dtype=np.float32).reshape(1, -1)

        # 중복된 이미지인지 확인
        duplicate_index = find_duplicate_image(image_vector)
        if duplicate_index is not None:
            st.info(
                f"{file.name} 이미지는 이미 업로드된 이미지와 동일합니다. 기존 이미지를 유지합니다."
            )
        else:
            # FAISS에 벡터 추가
            st.session_state.vector_db.add(image_vector)

            # 이미지 정보를 저장 (Base64 인코딩 포함)
            image_base64 = encode_image_to_base64(buffer.getvalue())
            st.session_state.image_metadata[st.session_state.vector_db.ntotal - 1] = {
                "name": file.name,
                "base64": image_base64,
            }

        st.image(image, caption=f"업로드된 이미지: {file.name}", use_column_width=True)

    st.success(
        f"{len(uploaded_files)}개의 이미지가 업로드되었습니다. 질문을 입력하세요!"
    )

# 기존 대화 기록 표시 (채팅 기록 유지)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 질문 처리
prompt = st.chat_input("이미지에 대해 질문을 입력하세요 (예: 공통점은 무엇인가요?)")

if prompt:
    # 사용자 입력 저장 및 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 업로드된 모든 이미지 정보를 OpenAI에게 전달 (Base64 포함)
    if len(st.session_state.image_metadata) > 0:
        image_info = []
        for i, idx in enumerate(st.session_state.image_metadata):
            image_data = st.session_state.image_metadata[idx]
            image_info.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data['base64']}"
                    },
                }
            )
        image_text = "\n".join(
            [
                f"이미지 {i+1}: {st.session_state.image_metadata[idx]['name']}"
                for i, idx in enumerate(st.session_state.image_metadata)
            ]
        )
    else:
        image_info = []
        image_text = "현재 업로드된 이미지가 없습니다."

    print(image_text)

    # OpenAI 프롬프트 구성
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"사용자가 {len(st.session_state.image_metadata)} 개의 이미지를 업로드했습니다.\n{image_text}\n\n사용자의 질문: {prompt}\n",
            },
        ]
        + image_info,  # 이미지 리스트를 추가
    )

    with st.chat_message("assistant"):
        st.markdown("응답 생성 중...")

        # OpenAI API 호출
        result = chat_model.invoke([message])
        response = result.content.strip()

        # AI 응답 저장 및 표시
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
