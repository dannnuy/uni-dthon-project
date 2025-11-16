# 파일명: model.py

import os
import torch
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

class QueryBasedDetector:
    def __init__(self, best_pt_path, device=None):
        """
        추론에 필요한 모든 모델(YOLO, CLIP, OCR)을
        한 번만 로드하여 클래스에 저장합니다.
        
        Args:
            best_pt_path (str): train.py로 학습시킨 'best.pt' 파일 경로
            device (str, optional): 'cuda' 또는 'cpu'. None이면 자동 감지.
        """
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")

        # 1. (Stage 1) YOLO 탐지기 로드
        self.detector = YOLO(best_pt_path).to(self.device)
        print(f"YOLO detector loaded from: {best_pt_path}")

        # 2. (Stage 2) CLIP 매칭기 로드
        clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        print(f"CLIP model loaded: {clip_model_name}")

        # 3. (Stage 2) OCR 로드 (캡션 추출용)
        self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=(self.device == "cuda"))
        print("EasyOCR loaded.")

        # 4. 스코어링 가중치 (실험적으로 조절)
        self.caption_weight = 0.7
        self.visual_weight = 0.3

    @torch.no_grad() # 추론 모드에서는 그래디언트 계산 비활성화
    def predict(self, image_path, query_text):
        """
        하나의 이미지와 질문(query)을 받아
        가장 점수가 높은 객체의 [x, y, w, h]를 반환합니다.
        
        Returns:
            list: [x, y, w, h] 형식의 BBox. 못 찾으면 [0, 0, 0, 0].
        """
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return [0, 0, 0, 0] # 이미지 로드 실패

        # --- 1. (Stage 1) 모든 표/차트 후보 탐지 ---
        yolo_results = self.detector.predict(image_pil, verbose=False)
        candidate_boxes = yolo_results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]

        if len(candidate_boxes) == 0:
            return [0, 0, 0, 0] # 탐지된 객체가 없음

        # --- 2. (Stage 2) 최고 점수 후보 매칭 ---
        
        # 2-1. 질문(Query) 텍스트를 CLIP 피처로 변환 (한 번만)
        query_inputs = self.clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        query_features = self.clip_model.get_text_features(**query_inputs).detach()

        best_box = None
        highest_score = -float('inf')

        for box in candidate_boxes:
            x1, y1, x2, y2 = map(int, box)

            # 2-2. 캡션 텍스트 추출 (Heuristic)
            caption_text = self._get_caption_text(image_pil, box)

            # 2-3. 이미지 조각(patch) 추출
            patch_img = image_pil.crop((x1, y1, x2, y2))

            # 2-4. (CLIP Encode) 이미지 조각과 캡션 텍스트를 피처로 변환
            inputs = self.clip_processor(
                text=[caption_text],
                images=[patch_img], 
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).to(self.device)
            
            visual_features = self.clip_model.get_image_features(inputs.pixel_values).detach()
            caption_features = self.clip_model.get_text_features(inputs.input_ids, inputs.attention_mask).detach()

            # 2-5. (Score) 질문(Query)과의 유사도 계산
            score_caption = torch.nn.functional.cosine_similarity(query_features, caption_features)
            score_visual = torch.nn.functional.cosine_similarity(query_features, visual_features)

            if not caption_text:
            # 캡션이 없으면 시각적 유사도만 사용
                w_caption = 0.0
                w_visual = 1.0
            else:
                # 캡션이 있으면 기존 가중치 사용
                w_caption = self.caption_weight
                w_visual = self.visual_weight
                
            final_score = (self.caption_weight * score_caption) + (self.visual_weight * score_visual)
            
            if final_score > highest_score:
                highest_score = final_score
                best_box = box # [x1, y1, x2, y2]

        # 3. 포맷 변환 및 반환
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            # [x1, y1, x2, y2] -> [x, y, w, h]로 변환
            pred_x = x1
            pred_y = y1
            pred_w = x2 - x1
            pred_h = y2 - y1
            return [float(pred_x), float(pred_y), float(pred_w), float(pred_h)]
        else:
            return [0, 0, 0, 0] # 매칭된 박스가 없음

    def _get_caption_text(self, image_pil, box):
        """(Helper) BBox 주변에서 캡션 텍스트를 OCR로 추출"""
        try:
            x1, y1, x2, y2 = map(int, box)
            img_width, img_height = image_pil.size
            
            # (Heuristic) 박스 바로 아래 50px 영역을 캡션 영역으로 가정
            cap_y1 = min(y2, img_height)
            cap_y2 = min(y2 + 50, img_height) 
            
            # 박스 너비와 동일하게 캡션 영역 자르기
            cap_x1 = max(0, x1)
            cap_x2 = min(x2, img_width)

            if cap_x1 >= cap_x2 or cap_y1 >= cap_y2:
                return "" # 영역이 없음

            caption_zone_img = image_pil.crop((cap_x1, cap_y1, cap_x2, cap_y2))
            
            # OCR 수행 (PIL 이미지가 아닌 numpy 배열을 받음)
            ocr_results = self.ocr_reader.readtext(np.array(caption_zone_img), detail=0)
            return " ".join(ocr_results)
        except Exception as e:
            # OCR 과정에서 에러가 나도 무시하고 빈 텍스트 반환
            print(f"Warning: OCR failed for box {box}. Error: {e}")
            return ""

# ---
# 이 파일이 직접 실행될 때 (테스트용)
if __name__ == "__main__":
    # train.py에서 생성된 'best.pt' 경로
    BEST_PT_FILE = "/data/danielsohn0827000/uni-dthon-project/best.pt"
    
    # 모델 로드
    model = QueryBasedDetector(best_pt_path=BEST_PT_FILE)
    
    # 임의의 테스트 이미지와 질문으로 테스트
    # (경로는 예시입니다. 실제 test 이미지/질문으로 바꿔보세요)
    test_img = "/data/danielsohn0827000/unid/train_valid/train/report_jpg/MI3_240819_TY1_0012_3.jpg"
    test_query = "유가 및 나프타 가격 꺾은선형"
    
    print(f"Test Query: {test_query}")
    bbox_xywh = model.predict(test_img, test_query)
    
    print(f"Predicted BBox [x, y, w, h]: {bbox_xywh}")