import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio

# 모델과 프로세서 로드
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 오디오 파일 로드
speech, sample_rate = torchaudio.load("0002.mp3")

# 필요한 경우 리샘플링
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    speech = resampler(speech)

# 모델 입력 준비
input_values = processor(
    speech.squeeze(), sampling_rate=16000, return_tensors="pt"
).input_values

# 모델 추론
with torch.no_grad():
    logits = model(input_values).logits

# 예측된 토큰 ID 추출
predicted_ids = torch.argmax(logits, dim=-1)

# 텍스트로 디코딩
transcription = processor.batch_decode(predicted_ids)[0]
print("Transcription:", transcription)
