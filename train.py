# 파일명: model.py
# (CUDA 'device-side assert' 오류 수정 완료)

import os
import torch
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO
# 👇 'AutoConfig'를 추가로 임포트합니다.
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor, AutoTokenizer, AutoConfig

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

        # --- 👇 [수정됨] CLIP 로더 로직 (CUDA 오류 해결) ---
        
        # 2. (Stage 2) CLIP 매칭기 로드
        clip_model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        base_clip_name = "openai/clip-vit-base-patch32" # 이미지 프로세서 참조용
        
        print(f"Loading CLIP model: {clip_model_name}")

        try:
            # (a) 로드할 모델의 '설정(Config)'을 명시적으로 불러옵니다.
            # 이 config에는 올바른 vocab_size(119547)가 포함되어 있습니다.
            config = AutoConfig.from_pretrained(clip_model_name)
            
            # (b) 모델 가중치 로드 (중요: config=config 전달)
            # 모델이 생성될 때 위에서 로드한 config를 사용하도록 강제합니다.
            self.clip_model = CLIPModel.from_pretrained(
                clip_model_name,
                config=config 
            ).to(self.device)
            
            # (c) 텍스트 토크나이저 로드 (다국어)
            tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
            
            # (d) 이미지 프로세서 로드 (원본)
            image_processor = CLIPImageProcessor.from_pretrained(base_clip_name)

            # (e) CLIPProcessor 수동 조합
            self.clip_processor = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)
            
            print(f"CLIP model and processor loaded successfully.")
            
        except Exception as e:
            print(f"Error loading CLIP components: {e}")
            raise e # 오류 발생 시 중지

        # --- 👆 [수정] 여기까지 ---

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
        """
        
        try:
            # 쿼리 텍스트가 None이거나 float(NaN)일 경우를 대비
            if not isinstance(query_text, str):
                query_text = "" # 빈 문자열로 처리

            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return [0, 0, 0, 0] 

        # [수정] YOLO conf 값 조절로 후보군 확대
        yolo_results = self.detector.predict(image_pil, verbose=False, conf=0.1) 
        candidate_boxes = yolo_results[0].boxes.xyxy.cpu().numpy() 

        if len(candidate_boxes) == 0:
            return [0, 0, 0, 0] 

        # 2-1. 질문(Query) 텍스트를 CLIP 피처로 변환 (한 번만)
        query_inputs = self.clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # (오류가 발생했던 지점)
        query_features = self.clip_model.get_text_features(**query_inputs).detach() # Shape: [1, D]

        best_box = None
        highest_score = -float('inf')

        for box in candidate_boxes:
            x1, y1, x2, y2 = map(int, box)

            # 1. 캡션 3개 추출 (위/아래)
            caption_text_above = self._get_caption_text(image_pil, box, "above")
            caption_text_below = self._get_caption_text(image_pil, box, "below")
            
            # 2-3. 이미지 조각(patch) 추출
            patch_img = image_pil.crop((x1, y1, x2, y2))
            
            # (a) 이미지 인코딩
            image_inputs = self.clip_processor(images=[patch_img], return_tensors="pt").to(self.device)
            visual_features = self.clip_model.get_image_features(image_inputs.pixel_values).detach() # Shape: [1, D]
            
            # (b) 텍스트 인코딩 (위/아래)
            text_inputs = self.clip_processor(
                text=[caption_text_above, caption_text_below],
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).to(self.device)
            text_features = self.clip_model.get_text_features(text_inputs.input_ids, text_inputs.attention_mask).detach() # Shape: [2, D]
            caption_features_above = text_features[0] # Shape: [D]
            caption_features_below = text_features[1] # Shape: [D]

            # --- [수정됨] 적응형 스코어링 로직 ---

            # (a) 이미지 점수 (공통)
            score_visual = torch.nn.functional.cosine_similarity(query_features, visual_features)

            # (b) 텍스트 점수 (OCR 결과가 있을 때만 계산)
            has_caption_above = len(caption_text_above.strip()) > 0
            has_caption_below = len(caption_text_below.strip()) > 0

            score_caption = 0.0 # 기본값

            if has_caption_above or has_caption_below:
                score_caption_above = 0.0
                score_caption_below = 0.0
                
                if has_caption_above:
                    score_caption_above = torch.nn.functional.cosine_similarity(query_features, caption_features_above.unsqueeze(0))
                if has_caption_below:
                    score_caption_below = torch.nn.functional.cosine_similarity(query_features, caption_features_below.unsqueeze(0))
                
                score_caption = max(score_caption_above, score_caption_below)
                
                # 캡션이 있을 때의 최종 점수
                final_score = (self.caption_weight * score_caption) + (self.visual_weight * score_visual)
            
            else:
                # 캡션이 아예 없으면 비주얼 점수만 100% 반영
                final_score = score_visual 
            
            # --- 👆 ---
            
            if final_score > highest_score:
                highest_score = final_score
                best_box = box 

        # 3. 포맷 변환 및 반환
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            pred_x = x1
            pred_y = y1
            pred_w = x2 - x1
            pred_h = y2 - y1
            return [float(pred_x), float(pred_y), float(pred_w), float(pred_h)]
        else:
            return [0, 0, 0, 0] 

    # --- [수정됨] OCR 영역 확장 (X축) ---
    def _get_caption_text(self, image_pil, box, position="below", margin_px=50, x_expand_px=100):
        """
        (Helper) BBox의 '위(above)' 또는 '아래(below)'에서 텍스트를 OCR로 추출
        (x_expand_px를 추가하여 BBox 너비보다 넓게 탐색)
        """
        try:
            x1, y1, x2, y2 = map(int, box)
            img_width, img_height = image_pil.size
            
            # BBox 너비보다 좌우 100px씩 넓게 탐색
            cap_x1 = max(0, x1 - x_expand_px)
            cap_x2 = min(img_width, x2 + x_expand_px)
            
            cap_y1, cap_y2 = 0, 0

            if position == "above":
                cap_y2 = max(0, y1)            
                cap_y1 = max(0, y1 - margin_px)
            else: # "below" (default)
                cap_y1 = min(y2, img_height)      
                cap_y2 = min(img_height, y2 + margin_px) 

            if cap_x1 >= cap_x2 or cap_y1 >= cap_y2:
                return "" # 영역이 없음

            caption_zone_img = image_pil.crop((cap_x1, cap_y1, cap_x2, cap_y2))
            
            ocr_results = self.ocr_reader.readtext(np.array(caption_zone_img), detail=0)
            return " ".join(ocr_results)
        except Exception as e:
            # OCR 에러가 나도 빈 문자열을 반환하여 메인 로직이 멈추지 않게 함
            # print(f"Warning: OCR failed for box {box} ({position}). Error: {e}")
            return ""
    # --- 👆 ---

# ---
# 이 파일이 직접 실행될 때 (테스트용)
if __name__ == "__main__":
    # train.py에서 생성된 'best.pt' 경로
    BEST_PT_FILE = "/data/danielsohn0827000/uni-dthon-project/best.pt"
    
    # 모델 로드
    model = QueryBasedDetector(best_pt_path=BEST_PT_FILE)
    
    # 임의의 테스트 이미지와 질문으로 테스트
    test_img = "/data/danielsohn0827000/unid/open/test/images/MI2_240819_TY1_0012_3.jpg" # 테스트 이미지 경로로 수정
    test_query = "유가 및 나프타 가격 꺾은선형"
    
    print(f"Test Query: {test_query}")
    bbox_xywh = model.predict(test_img, test_query)
    
    print(f"Predicted BBox [x, y, w, h]: {bbox_xywh}")