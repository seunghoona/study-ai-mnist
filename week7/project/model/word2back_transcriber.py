import os
import torch
import yaml
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
from pyannote.audio import Pipeline
from common.logger_config import setup_logger
from common.file_manager import FileManager

logger = setup_logger()


class Word2backTranscriber:
    """음성을 텍스트로 변환하고 요약하는 클래스"""

    def __init__(self, hf_token, base_dir="people"):
        logger.info("Transcriber 초기화 시작")
        try:
            if not hf_token:
                logger.error("Hugging Face 토큰이 설정되지 않았습니다")
                raise ValueError("Hugging Face 토큰이 필요합니다.")

            with open("config.yml", "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                logger.debug("설정 파일 로드 완료")

            self.file_manager = FileManager(base_dir)

            # Wav2Vec2 모델 로드
            self.processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

            self.model.to(torch.device("mps" if torch.mps.is_available() else "cpu"))
            logger.debug("Wav2Vec2 모델 로드 완료")

            # 화자 분리 파이프라인 로드
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
            )
            self.pipeline.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            logger.info("화자 분리 파이프라인 초기화 완료")

            self.is_split = False
            self.transcripts = []

        except Exception as e:
            logger.error(f"초기화 중 오류 발생: {str(e)}")
            raise

    def _convert_audio(self, file_name):
        """오디오 파일을 변환하고 필요하면 분할하는 함수"""
        logger.info(f"오디오 파일 변환 시작: {file_name}")
        try:
            file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
            logger.debug(f"파일 크기: {file_size_mb:.2f}MB")

            if file_size_mb > self.config["transcriber"]["audio"]["target_size_mb"]:
                logger.info("파일 분할 시작")
                audio = AudioSegment.from_file(file_name)
                output_dir = self.file_manager.get_file_path("split_chunks", "")
                os.makedirs(output_dir, exist_ok=True)

                split_length = (
                    self.config["transcriber"]["audio"]["split_length_min"] * 60 * 1000
                )
                for i in range(0, len(audio), split_length):
                    chunk = audio[i : i + split_length]
                    chunk_name = os.path.join(
                        output_dir, f"chunk_{i // split_length}.mp3"
                    )
                    chunk.export(chunk_name, format="mp3")
                self.is_split = True
                logger.info(
                    f"파일 분할 완료: {len(os.listdir(output_dir))}개 청크 생성"
                )
            else:
                self.is_split = False

        except Exception as e:
            logger.error(f"오디오 변환 중 오류 발생: {str(e)}")
            raise

    def transcribe(self, file_name):
        """음성을 텍스트로 변환하는 메소드"""
        logger.info(f"음성 변환 시작: {file_name}")
        try:
            self._convert_audio(file_name)
            if self.is_split:
                split_dir = self.file_manager.get_file_path("split_chunks", "")
                segments = []
                for chunk in sorted(os.listdir(split_dir)):
                    if chunk.endswith(".mp3"):
                        chunk_path = os.path.join(split_dir, chunk)
                        segments.extend(self._transcribe_audio(chunk_path))
            else:
                segments = self._transcribe_audio(file_name)

            logger.info("화자 분리 시작")
            diarization = self._diarize_speaker(file_name)
            self.transcripts = self._match_segments_with_speakers(segments, diarization)

            logger.info(f"변환 완료: {len(self.transcripts)}개 세그먼트 생성")
            return self.transcripts
        except Exception as e:
            logger.error(f"변환 중 오류 발생: {str(e)}")
            raise

    def _transcribe_audio(self, file_name):
        """Wav2Vec2를 이용한 오디오 변환"""
        logger.info(f"Wav2Vec2를 이용한 변환 시작: {file_name}")

        # 오디오 파일 로드
        speech_array, sample_rate = torchaudio.load(file_name)
        logger.debug(
            f"원본 샘플링 레이트: {sample_rate}, 데이터 크기: {speech_array.shape}"
        )

        # 샘플링 레이트 변환 (16kHz가 아닐 경우 변환)
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sample_rate
            )
            speech_array = resampler(speech_array).squeeze()
            logger.debug(f"샘플링 레이트 변환 후 데이터 크기: {speech_array.shape}")
        else:
            speech_array = speech_array.squeeze()

        # 데이터 전처리
        speech_array = speech_array.numpy()
        inputs = self.processor(
            speech_array,
            sampling_rate=target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 모델 추론
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # 결과 해석
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        logger.debug(f"변환된 텍스트 길이: {len(transcription)}")

        return [
            {
                "start": 0,
                "end": len(speech_array) / target_sample_rate,
                "text": transcription,
            }
        ]

    def _diarize_speaker(self, file_name):
        """화자 분리 수행"""
        diar = self.pipeline(file_name)
        return diar.to_lab()

    def _match_segments_with_speakers(self, segments, diarization):
        """변환된 텍스트와 화자 정보 매칭"""
        matched_segments = []
        for diar in diarization.split("\n"):
            try:
                start_time, end_time, speaker = diar.split(" ")
                start_time, end_time = float(start_time), float(end_time)
                matched_segments.append(
                    {"start": start_time, "end": end_time, "label": speaker, "text": ""}
                )
            except ValueError:
                continue

        for segment in segments:
            for diar in matched_segments:
                if diar["start"] <= segment["start"] <= diar["end"]:
                    diar["text"] += segment["text"] + " "

        return [item for item in matched_segments if item["text"]]
