import streamlit as st
from streamlit_option_menu import option_menu
from file_manager import FileManager
from transcriber import Transcriber
import yaml
from constants import FileType, FileExtension, MimeType

# 설정 파일 로드
with open("config.yml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 기본 디렉터리 설정
SAVE_DIR = config['paths']['save_dir']
file_manager = FileManager(SAVE_DIR)
transcriber = Transcriber()

# UI 설정
st.set_page_config(
    page_title=config['app']['title'], 
    page_icon=config['app']['icon'], 
    layout="wide"
)
st.title(config['app']['title'])
st.write(config['app']['description'])

# 파일 업로드
sound_file = st.file_uploader(
    "음성 파일을 업로드하세요", 
    type=config['file']['allowed_audio_extensions'], 
    disabled=False
)

# 사이드바 (노트 목록)
with st.sidebar:
    notes_list = [config['ui']['new_note_label']] + file_manager.get_note_list()
    selected_note = option_menu("노트 목록", notes_list, menu_icon="book", default_index=0)

# 기존 노트 선택 시 파일 목록 로드
if selected_note != config['ui']['new_note_label']:
    transcript_file = file_manager.get_file_path(selected_note, f"{selected_note}{FileExtension.JSONL.value}")
    transcript_data = file_manager.load_jsonl(transcript_file)
else:
    transcript_data = []

# 📌 오디오 파일 저장 및 변환
if sound_file:
    st.audio(sound_file, format="audio/wav")
    
    file_name = sound_file.name
    file_path = file_manager.get_file_path(selected_note, file_name)
    
    if not file_manager.load_file(file_path):
        with open(file_path, "wb") as f:
            f.write(sound_file.read())

    col1, col2 = st.columns(2)
    if col1.button("변환 시작"):
        with st.spinner("변환 중..."):
            transcript_data = transcriber.transcribe()
            transcript_file = file_manager.get_file_path(selected_note, f"{selected_note}{FileExtension.JSONL.value}")
            file_manager.save_jsonl(transcript_file, transcript_data)

    if transcript_data:
        st.write("변환된 텍스트:")
        for item in transcript_data:
            st.text_area(
                f"{item['label']} ({item['start']} - {item['end']})", 
                item["text"], 
                height=config['ui']['default_text_height']
            )
        
        formatted_text = "\n".join(f"{t['label']}: {t['text']}" for t in transcript_data)
        st.download_button("변환된 텍스트 다운로드", formatted_text, file_name="transcript.txt")

    if col2.button("요약"):
        with st.spinner("요약 중..."):
            summary_text = transcriber.summarize()
            summary_file = file_manager.get_file_path(selected_note, f"{selected_note}_summary.txt")
            file_manager.save_file(summary_file, summary_text)
            st.write("요약 결과:")
            st.text_area("요약", summary_text, height=config['ui']['default_text_height'])

            st.download_button("요약 다운로드", summary_text, file_name="summary.txt")

# 🔹 오디오 파일 다운로드 (바이너리 처리)
if selected_note != config['ui']['new_note_label']:
    sound_path = file_manager.get_file_path(selected_note, file_name)
    sound_data = file_manager.load_file(sound_path)

    if isinstance(sound_data, bytes):  # 바이너리 파일인 경우
        st.download_button(
            "오디오 파일 다운로드",
            sound_data,
            file_name=file_name,
            mime=MimeType.AUDIO.value,
        )