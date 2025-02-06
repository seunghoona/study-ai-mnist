import logging
import os
from datetime import datetime


def setup_logger():
    # 로그 디렉토리 생성
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 로그 파일명에 날짜 포함
    log_file = os.path.join(
        log_dir, f"speechnote_{datetime.now().strftime('%Y%m%d')}.log"
    )

    # 로거 설정
    logger = logging.getLogger("SpeechNote")
    logger.setLevel(logging.DEBUG)

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 포맷 설정
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
