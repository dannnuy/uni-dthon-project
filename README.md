Uni-DTHON 데이터톤 트랙 제출 모델

1. 프로젝트 개요 및 모델 아키텍처

본 프로젝트는 문서 이미지와 자연어 질의를 입력받아 질의와 의미적으로 관련된 시각 요소(표/차트)의 위치(Bounding Box)를 예측하는 질의 기반 비전-언어 모델을 개발합니다.

1-1. 2-Stage 모델 구조 (Two-Stage Architecture)

규칙에 허용된 사전 학습 모델(CLIP)을 활용하여, 탐지(Detection)와 매칭(Ranking) 역할을 분리한 2단계 파이프라인을 구축했습니다.

Stage 1: 후보 탐지기 (YOLOv8 Detector)

역할: 이미지 내에 존재하는 모든 '표' 및 '차트' 영역의 Bounding Box를 정확하게 찾아냅니다.

훈련: 제공된 train_valid 데이터의 레이아웃 정보(..._json의 bounding_box)를 이용하여 YOLOv8n 모델을 Fine-tuning했습니다.

Stage 2: 쿼리 기반 랭커 (CLIP Ranker)

역할: Stage 1에서 탐지된 여러 개의 후보 BBox와 사용자의 질의 텍스트를 입력받아, 두 요소 간의 **코사인 유사도(Cosine Similarity)**를 계산합니다.

선택: 유사도 점수가 가장 높은 BBox를 최종 정답으로 선택합니다.

1-2. 코드 구성

파일명

역할

requirements.txt

파이썬 라이브러리 의존성 목록 (설치 필수)

preprocess.py

train_valid JSON 데이터를 YOLOv8 학습에 필요한 포맷(.txt, data.yaml)으로 변환

train.py

전처리된 데이터를 이용하여 YOLOv8 모델(Stage 1)을 파인튜닝

model.py

YOLO와 CLIP 모델을 통합한 DetectionRanker 클래스 정의 (2-Stage 로직 구현)

test.py

훈련된 best.pt와 test 데이터를 사용하여 최종 submission.csv를 생성

2. 학습 및 추론 환경

2-1. 학습 환경 (Reproduce Environment)

항목

설정

참고

GPU

NVIDIA A100 또는 RTX 3090 (1장)

스크립트는 --gres=gpu:1로 요청됨

CPU

8 Cores

SLURM 설정 (--cpus-per-gpu=8)

RAM

32GB

SLURM 설정 (--mem-per-gpu=32G)

Batch Size

32

GPU 메모리 환경에 따라 조정 가능

모델 (Stage 1)

YOLOv8n (Fine-tuned)

yolov8m.pt 등으로 업그레이드 가능

모델 (Stage 2)

CLIP ViT-Base-Patch32

clip-vit-large-patch14로 업그레이드 가능

2-2. 재현 및 실행 방법 (MANDATORY)

모든 스크립트는 프로젝트 루트 디렉토리(/data/danielsohn0827000/uni-dthon-project)에서 실행됩니다.

Step 0: 데이터 및 환경 설정

데이터 압축 해제: train_valid.zip, train_valid.zip.1, open.zip 파일을 압축 해제하고, train_valid 폴더와 test 폴더가 /data/danielsohn0827000/uni/ 경로에 있는지 확인합니다.

의존성 설치:

pip install -r requirements.txt


Step 1: 데이터 전처리 (Preprocessing)

제공된 JSON을 YOLO 학습에 필요한 이미지/라벨 포맷으로 변환합니다.

python3 preprocess.py


(실행 결과: /data/danielsohn0827000/uni-dthon-project/yolo_dataset 폴더 및 data.yaml 파일 생성)

Step 2: 모델 학습 (YOLO Fine-tuning)

YOLOv8 모델을 50 Epochs 동안 파인튜닝합니다.

특징: train.py는 runs/finetune_run/weights/last.pt 체크포인트가 존재하면 자동으로 학습을 재개합니다.

# SLURM 배치 잡 제출 (권장)
sbatch unid.sh

# 또는 GPU 할당 후 직접 실행 (빠른 테스트)
python3 train.py


(최종 결과: /data/danielsohn0827000/uni-dthon-project/best.pt 파일에 가장 성능이 좋은 가중치가 복사됨)

Step 3: 추론 및 제출 파일 생성 (Inference)

훈련된 best.pt 파일과 CLIP을 사용하여 test 데이터에 대한 예측을 수행하고 submission.csv를 생성합니다.

# GPU 할당 후 실행 필요 (srun 또는 sbatch 필요)
python3 test.py


(최종 결과: 프로젝트 루트에 submission.csv 파일 생성)

3. 성능 개선 전략 (0.60 점수 개선 방안)

초기 제출 점수(0.60)가 낮은 경우, 다음 개선 전략을 순차적으로 적용하여 성능을 높일 수 있습니다.

훈련 횟수 확보: EPOCHS를 50회 이상으로 충분히 늘려 모델이 val/mAP를 최대치까지 달성하도록 합니다. (현재 14 Epochs에서 멈췄다면, 남은 Epochs를 반드시 완료해야 합니다.)

YOLO 모델 업그레이드: train.py에서 YOLO("yolov8n.pt") 대신 YOLO("yolov8m.pt") 또는 YOLO("yolov8l.pt")를 사용하여 모델의 일반화 능력을 향상시킵니다.

CLIP 랭커 파워 업그레이드: model.py 파일의 CLIP_MODEL_ID를 **openai/clip-vit-large-patch14**와 같은 더 강력한 모델로 변경하여 쿼리-이미지 매칭의 정확도를 높입니다.