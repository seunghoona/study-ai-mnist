from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_id = "google/gemma-2b-it"


def get_model():
    hf_token = os.getenv("HF_TOKEN")  # Hugging Face Access Token
    login(hf_token)  # Hugging Face 로그인

    model_id = "google/gemma-2b-it"  # 모델 ID

    # 모델과 토크나이저 로드 (MPS 지원)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("mps")  # MPS에 로드

    # HuggingFacePipeline 초기화
    llm = HuggingFacePipeline(
        pipeline=pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device="mps",  # MPS 설정
            max_new_tokens=256,  # 최대 토큰 생성 수
            do_sample=False,  # Deterministic 설정
        )
    )

    # ChatHuggingFace로 Wrapping
    model = ChatHuggingFace(llm=llm)
    return model
