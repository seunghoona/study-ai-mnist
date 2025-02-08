import os
import torch
import yaml
from openai import OpenAI
from pydub import AudioSegment
from pyannote.audio import Pipeline
from common.logger_config import setup_logger
from common.file_manager import FileManager

logger = setup_logger()

class Transcriber:
    """음성을 텍스트로 변환하고 요약하는 클래스"""

    def __init__(self, openai_api_key, hf_token, base_dir="upload_people"):
        logger.info("Transcriber 초기화 시작")
        try:

            if not openai_api_key or not hf_token:
                logger.error("API 키가 설정되지 않았습니다")
                raise ValueError(
                    "API 키가 설정되지 않았습니다. .env 파일을 확인해주세요."
                )

            # 설정 파일 로드
            with open("config.yml", "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                logger.debug("설정 파일 로드 완료")

            # FileManager 초기화
            self.file_manager = FileManager(base_dir)  # FileManager 인스턴스 생성

            # OpenAI 클라이언트 초기화
            self.client = OpenAI(api_key=openai_api_key)
            logger.debug("OpenAI 클라이언트 초기화 완료")

            # 화자 분리 파이프라인 초기화
            logger.info("화자 분리 파이프라인 초기화 시작")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            self.pipeline.to(torch.device("mps" if torch.mps.is_available() else "cpu"))
            logger.info("화자 분리 파이프라인 초기화 완료")

            self.is_split = False
            self.transcripts = []
            self.summary = ""

        except Exception as e:
            logger.error(f"초기화 중 오류 발생: {str(e)}")
            raise

    def _convert_audio(self, file_name):
        logger.info(f"오디오 파일 변환 시작: {file_name}")
        try:
            logger.info(os.path)
            file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
            logger.debug(f"파일 크기: {file_size_mb:.2f}MB")

            if file_size_mb > self.config["transcriber"]["audio"]["target_size_mb"]:
                logger.info("파일 분할 시작")
                audio = AudioSegment.from_file(file_name)

                # FileManager를 사용하여 경로 생성
                output_dir = self.file_manager.get_file_path(
                    "split_chunks", ""
                )  # "split_chunks" 폴더 경로
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
                logger.debug("파일 분할 불필요")
                self.is_split = False

        except Exception as e:
            logger.error(f"오디오 변환 중 오류 발생: {str(e)}")
            raise

    def _diarize_speaker(self, file_name, speaker_num=2):
        """화자 분리 수행"""
        file_dir, file_ext = os.path.splitext(file_name)
        if file_ext == ".wav":
            wav_file = file_name
        else:
            audio = AudioSegment.from_file(file_name)
            wav_file = f"{file_dir}.wav"
            audio.export(wav_file, format="wav")

        diar = self.pipeline(wav_file, num_speakers=speaker_num)
        speaker = diar.to_lab()

        if file_ext != ".wav":
            os.remove(wav_file)
        return speaker

    def _compress_diarization(self, segments_diar):
        """화자 분리 결과 압축"""
        compressed_diar = []
        segments_diar = segments_diar.split("\n")
        for segment in segments_diar:
            try:
                start_time, end_time, speaker = segment.split(" ")
            except ValueError:
                break

            start_time = float(start_time)
            end_time = float(end_time)

            if not compressed_diar or compressed_diar[-1]["label"] != speaker:
                compressed_diar.append(
                    {"start": start_time, "end": end_time, "label": speaker}
                )
            else:
                compressed_diar[-1]["end"] = end_time

        return compressed_diar

    def transcribe(self, file_name, speaker_num=2):
        logger.info(f"음성 변환 시작: {file_name}")
        try:
            self._convert_audio(file_name)

            if self.is_split:
                logger.info("분할된 파일 처리 시작")
                split_dir = self.file_manager.get_file_path("split_chunks", "")
                segments = []
                for i, chunk in enumerate(sorted(os.listdir(split_dir))):
                    if chunk.endswith(".mp3"):
                        chunk_path = os.path.join(split_dir, chunk)
                        with open(chunk_path, "rb") as audio_file:
                            response = self.client.audio.transcriptions.create(
                                file=audio_file,
                                model=self.config["transcriber"]["openai"][
                                    "transcript_model"
                                ],
                                response_format="verbose_json",
                                timestamp_granularities=["segment"],
                            )
                            for segment in response.segments:
                                segments.append(
                                    {
                                        "start": segment["start"]
                                        + (
                                            i
                                            * self.config["transcriber"]["audio"][
                                                "split_length_min"
                                            ]
                                        ),
                                        "end": segment["end"]
                                        + (
                                            i
                                            * self.config["transcriber"]["audio"][
                                                "split_length_min"
                                            ]
                                        ),
                                        "text": segment["text"],
                                    }
                                )
            else:
                logger.info("단일 파일 처리")
                with open(file_name, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        file=audio_file,
                        model=self.config["transcriber"]["openai"]["transcript_model"],
                        response_format="verbose_json",
                        timestamp_granularities=["segment", "word"],
                    )

                    segments = [
                        {"start": seg.start, "end": seg.end, "text": seg.text}
                        for seg in response.segments
                    ]

            logger.info("화자 분리 시작")
            diarization = self._diarize_speaker(file_name, speaker_num)
            compressed_diar = self._compress_diarization(diarization)

            logger.info("텍스트 매칭 시작")
            self.transcripts = self._match_segments_with_speakers(
                segments, compressed_diar
            )

            logger.info(f"변환 완료: {len(self.transcripts)}개 세그먼트 생성")
            return self.transcripts

        except Exception as e:
            logger.error(f"변환 중 오류 발생: {str(e)}")
            raise

    def _match_segments_with_speakers(self, segments, diarization):
        """변환된 텍스트와 화자 정보 매칭"""
        matched_segments = [
            {
                "start": diar["start"],
                "end": diar["end"],
                "label": diar["label"],
                "text": "",
            }
            for diar in diarization
        ]

        for segment in segments:
            max_overlap = 0
            max_overlap_idx = 0
            for idx, diar in enumerate(diarization):
                overlap = min(segment["end"], diar["end"]) - max(
                    segment["start"], diar["start"]
                )
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_overlap_idx = idx
            if max_overlap > 0:
                matched_segments[max_overlap_idx]["text"] += (
                    segment["text"].strip() + " "
                )

        return [item for item in matched_segments if item["text"]]

    def summarize(self):
        logger.info("요약 시작")
        try:
            if not self.transcripts:
                logger.warning("변환된 텍스트가 없습니다")
                return "변환된 텍스트가 없습니다."

            full_text = "\n".join(
                f"{t['label']}: {t['text']}" for t in self.transcripts
            )
            logger.debug(f"요약할 텍스트 길이: {len(full_text)} 문자")

            prompt = self.config["transcriber"]["summary_prompt"]

            response = self.client.chat.completions.create(
                model=self.model,  # 선택된 모델 사용
                messages=[{"role": "user", "content": f"{prompt}\n\n{full_text}"}],
            )

            self.summary = response.choices[0].message.content
            logger.info("요약 완료")
            return self.summary

        except Exception as e:
            logger.error(f"요약 중 오류 발생: {str(e)}")
            raise
