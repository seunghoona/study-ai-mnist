import time

class Transcriber:
    """음성을 텍스트로 변환하고 요약하는 클래스"""

    def __init__(self):
        self.transcripts = []
        self.summary = ""

    def transcribe(self):
        """음성을 텍스트로 변환 (모의 기능)"""
        time.sleep(3)  # 변환 시간 시뮬레이션
        self.transcripts = [
            {"label": "화자1", "start": 0.0, "end": 1.0, "text": "안녕하세요!"},
            {"label": "화자2", "start": 1.0, "end": 2.0, "text": "네, 반갑습니다."}
        ]
        return self.transcripts

    def summarize(self):
        """요약 기능 (모의 기능)"""
        time.sleep(3)  # 요약 시간 시뮬레이션
        self.summary = "화자1이 인사했고, 화자2가 반갑다고 응답함."
        return self.summary