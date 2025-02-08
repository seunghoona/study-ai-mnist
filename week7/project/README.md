# 심화과제 제출 

- 심화과제의 요구사항대로 학습을 진행하지 못했습니다. 
- LLM 모델로 SFT 학습을 진행하고자 했지만 음성모델을 어떤 방식으로 학습해야할지 정하지 못하여 허깅페이스에서 was2vec 모델을 학습시켰습니다. 

- 음성 모델에 대해 이해하고 학습하는데 너무 시간을 많이 쏟아서 요약 부분을 완성 하지 못했습니다. 
- 요약 transcriber.py 안에 `summarize` 함수를 완성했다면 SFT 확인 해볼 수 있었을 것 같은데
우선 완성된 것 까지는 제출을 해보는게 좋을 것 같아서 제출했습니다. 
- study 폴더에 [corpus.json](./study/train_file//corpus.json) 음성 학습을 시키기 위한 데이터를 생성했습니다.
- train.ipynb 파일에 다른 음성 모델을 학습 처리습니다. [train.ipynb](./study/wav2vec2_finetuning/train.ipynb)



### 파일 목록

| **파일명**        | **경로**                 | **정의 및 형식** |
|------------------|-----------------------|----------------|
| [`main.py`](./main.py)       | `/main.py`       | Streamlit 애플리케이션 메인 파일 |
| [`transcriber.py`](./model/transcriber.py) | `model/transcriber.py` | 화자 정리 및 OpenAI API 호출 담당 클래스 |
| [`train.ipynb`](./study/wav2vec2_finetunig/train.ipynb)       | `/study/wav2vec2_finetunig/train.ipynb`       | 모델 학습 및 평가 스크립트 |
| [`config.yml`](./config.yml)    | `./config.yml` | 학습 파라미터 및 설정 파일 |
| [`file_manager.py`](./common/file_manager.py) | `/common/file_manager.py` | 파일 저장 및 불러오기 담당 클래스 |
| [`constants.py`](./common/constants.py)   | `/common/constants.py` | 상수 정의 |
| [`logger_config.py`](./common/logger_config.py) | `/common/logger_config.py` | 로깅 설정 |
| [`utils.py`](./common/utils.py)       | `/common/utils.py`     | 유틸리티 함수 모음 |


# 1. 프로젝트 정의


1. 프로젝트명 
- Clova 음성 대화 화자분리 및 스크립트 출력하기

2. 방법론 정하기
- 사용자의 상담 음성파일을 화자별 분리하여 출력을 처리하기 위해서 MLLM 모델인 Wishper를 사용한다. 

3. 실제로 prompt를 줬을 때 LLM의 응답
- 음성 내용을 상담자 와 내담자별로 출력한다. 
- 상담 내용의 질문 내용을 요약해서 가져온다. 

4. 음성 파일 스크립트는 현재 upload_people 폴더내 저장 
- [스크립트파일보기](./upload_people/상담1/상담1.jsonl)

# 2 실행 영상 

[실행영상다운받기](./etc/실행영상.mp4)
 
## 앞으로의 계획 

- [ ] 폴더 내 저장
- [ ] LLM 모델외 huggingface 에서 모델을 가져와서 Instruction-data 준비 해서 SFT 해보기 
    - [ ] 샘플링 맞추기
- [ ] RAG 를 사용하기 
- [ ] 코드 리팩토링
