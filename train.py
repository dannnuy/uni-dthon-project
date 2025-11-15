# 파일명: train.py

import os
import torch
import shutil
from ultralytics import YOLO

# --- 1. 경로 설정 ---

# preprocess.py의 OUTPUT_DATA_DIR에 생성된 data.yaml 경로
DATA_YAML_PATH = "/data/danielsohn0827000/uni/yolo_dataset/data.yaml"

# 다운로드한 사전 학습 가중치 경로
PRETRAINED_MODEL_PATH = "/data/danielsohn0827000/uni-dthon-project/report-8n.pt"

# 학습 결과(runs)가 저장될 '프로젝트' 디렉토리
OUTPUT_PROJECT_DIR = "/data/danielsohn0827000/uni-dthon-project"

# 최종 best.pt를 저장하고 싶은 '정확한' 파일 경로
FINAL_BEST_PT_PATH = "/data/danielsohn0827000/uni-dthon-project/best.pt"

# --- 2. 학습 설정 ---
EPOCHS = 50
BATCH_SIZE = 32
IMG_SIZE = 640
EXPERIMENT_NAME = "finetune_run" # 학습 결과가 저장될 하위 폴더 이름

# --- 3. (추가) 재개할 체크포인트 경로 설정 ---
# YOLO는 이 경로에 'last.pt'와 'best.pt'를 저장합니다.
LAST_PT_PATH = os.path.join(OUTPUT_PROJECT_DIR, EXPERIMENT_NAME, 'weights/last.pt')


def main():
    # 0. 필수 파일 및 경로 검사
    if not os.path.exists(DATA_YAML_PATH):
        print(f"오류: data.yaml 파일을 찾을 수 없습니다. 경로: {DATA_YAML_PATH}")
        print("preprocess.py를 먼저 실행했는지 확인하세요.")
        return

    # 1. 장치 설정
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"사용 장치: {device}")

    # --- 2. 모델 로드 (수정된 부분: 재개 기능) ---
    model = None
    train_args = {
        "data": DATA_YAML_PATH,
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMG_SIZE,
        "device": device,
        "project": OUTPUT_PROJECT_DIR,
        "name": EXPERIMENT_NAME,
        "exist_ok": True
    }

    if os.path.exists(LAST_PT_PATH):
        print(f"발견된 체크포인트: {LAST_PT_PATH}")
        print(">>> 이어서 학습을 재개합니다.")
        model = YOLO(LAST_PT_PATH)  # 'last.pt'에서 모델 로드
        train_args["resume"] = True  # True로 설정 시 옵티마이저 상태 등 복원
    else:
        print(f"체크포인트 없음. {PRETRAINED_MODEL_PATH}에서 새 학습을 시작합니다.")
        if not os.path.exists(PRETRAINED_MODEL_PATH):
            print(f"오류: 사전 학습 모델을 찾을 수 없습니다. 경로: {PRETRAINED_MODEL_PATH}")
            return
        model = YOLO(PRETRAINED_MODEL_PATH)  # 'report-8n.pt'에서 모델 로드
        # 'resume' 키는 추가되지 않음

    # 3. 모델 학습
    print(f"학습을 시작합니다...")
    # **train_args: 딕셔너리를 인수로 자동 분해 (resume=True가 있거나 없게 됨)
    results = model.train(**train_args)
    
    # 4. 학습 완료 및 best.pt 복사
    print("학습 완료.")
    
    # YOLO가 생성한 best.pt의 원본 경로
    original_best_pt_path = os.path.join(results.save_dir, 'weights/best.pt')

    if os.path.exists(original_best_pt_path):
        print(f"최종 'best.pt'를 요청한 경로로 복사합니다: {FINAL_BEST_PT_PATH}")
        shutil.copy(original_best_pt_path, FINAL_BEST_PT_PATH)
        print("복사 완료. 'best.pt' 파일을 추론(test.py)에 사용할 수 있습니다.")
    else:
        # best.pt는 검증 성능이 좋아질 때만 저장되므로, 
        # 학습이 일찍 끊기면(예: 1에포크) 아직 없을 수도 있음
        print(f"경고: 'best.pt' 원본 파일을 찾을 수 없습니다. 경로: {original_best_pt_path}")
        print("'best.pt'는 최소 1 epoch의 검증(val)이 끝나야 생성됩니다.")

if __name__ == "__main__":
    main()