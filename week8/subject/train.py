import torch
import time
import wandb
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer

# WandB 로그인 및 프로젝트 초기화
wandb.login()

# 모델 및 토크나이저 로드
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 로드 및 포맷팅
dataset_name = "sahil2801/CodeAlpaca-20k"
dataset = load_dataset(dataset_name)
# 포맷팅 및 토큰화 함수 정의
def formatting_prompts_func(batch):
    texts = [
        f"Instruction: {instr}\nInput: {inp}\nOutput: {out}"
        for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    
    # 여기서 토큰화 진행
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

# dataset을 변환할 때, `batched=True`를 설정
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset["train"].column_names)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# LoRA 실험 설정
lora_rs = [8, 128, 256]
lora_alpha = 32
lora_dropout = 0.1

# 결과 저장을 위한 리스트
results = []

for lora_r in lora_rs:
    print(f"Training with LoRA r={lora_r}")
    wandb.init(
      project="week8",
      name=f"LoRA-r{lora_r}",
      group="LoRA_Comparision",
      reinit=True
    )
    # LoRA Config 정의
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )
    
    # 모델 복사 및 LoRA 적용
    model_lora = get_peft_model(model, lora_config)
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=f"/tmp/clm-instruction-tuning-r{lora_r}",
        per_device_train_batch_size=4,
        save_steps=500,
        num_train_epochs=1,
        logging_dir=f"/tmp/logs-r{lora_r}",
        logging_steps=100,
        save_total_limit=1,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        max_steps=1500
    )
    
    # SFTTrainer 초기화 및 학습 시작 시간 기록
    trainer = SFTTrainer(
        model=model_lora,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collator,
    )
    
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    # 학습 속도 및 손실 값 저장
    training_time = end_time - start_time
    final_loss = train_result.training_loss
    
    # GPU 메모리 사용량 측정
    max_memory = round(torch.cuda.max_memory_allocated(0) / 1024**3, 1)
    
    print(f"Max Alloc: {max_memory} GB")
    
    # 결과 저장
    results.append({
        "lora_r": lora_r,
        "final_loss": final_loss,
        "training_time": training_time,
        "max_memory": max_memory
    })
    
    # WandB에 기록
    wandb.log({
        "lora_r": lora_r,
        "final_loss": final_loss,
        "training_time": training_time,
        "max_memory": max_memory
    })
    
    # 모델 저장
    model_lora.save_pretrained(f"/tmp/clm-instruction-tuning-r{lora_r}")

# WandB 종료
wandb.finish()