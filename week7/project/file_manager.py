import os
import json
import chardet  # 인코딩 자동 감지를 위한 라이브러리
import yaml


class FileManager:
    """파일 저장 및 불러오기 담당 클래스"""

    def __init__(self, base_dir):
        """기본 저장 디렉토리 설정"""
        with open("config.yml", "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.base_dir = os.path.abspath(base_dir)  # 절대 경로 변환
        os.makedirs(self.base_dir, exist_ok=True)

    def get_note_list(self):
        """저장된 노트 목록 반환"""
        return os.listdir(self.base_dir)

    def get_file_path(self, note_name, file_name):
        """노트 내 특정 파일 경로 반환"""
        note_path = os.path.abspath(
            os.path.join(self.base_dir, note_name)
        )  # 절대 경로 변환
        os.makedirs(note_path, exist_ok=True)
        return os.path.join(note_path, file_name)

    def save_file(self, file_path, data):
        """텍스트 파일 저장"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data)

    def load_file(self, file_path):
        """파일 불러오기 (바이너리 & 텍스트 구분)"""
        if not os.path.exists(file_path):
            return None

        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.config["file"]["binary_extensions"]:
            with open(file_path, "rb") as f:
                return f.read()

        # 텍스트 파일: 인코딩 감지 후 처리
        with open(file_path, "rb") as f:
            raw_data = f.read()  # 바이너리 데이터 읽기

        # chardet으로 인코딩 감지
        encoding_detected = chardet.detect(raw_data)["encoding"]

        if not encoding_detected:
            encoding_detected = "utf-8"  # 기본 UTF-8로 설정

        try:
            return raw_data.decode(encoding_detected)  # 감지된 인코딩 적용
        except UnicodeDecodeError:
            return raw_data.decode("ISO-8859-1", errors="replace")

    def save_jsonl(self, file_path, data_list):
        """JSONL 형식 데이터 저장 (개행 추가)"""
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data_list:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")  # JSON 데이터가 정확히 구분되도록 개행 추가

    def load_jsonl(self, file_path):
        """JSONL 형식 데이터 불러오기"""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        return []
