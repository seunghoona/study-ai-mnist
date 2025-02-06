def format_transcript(transcript_data):
    """텍스트 변환 데이터를 보기 좋게 정리"""
    return "\n".join(f"{item['label']}: {item['text']}" for item in transcript_data)