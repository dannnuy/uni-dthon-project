# 파일명: test.py

import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from model import QueryBasedDetector # 👈 model.py에서 클래스 임포트

# --- 1. 경로 설정 ---

# train.py가 생성한 최종 모델
BEST_PT_PATH = "/data/danielsohn0827000/uni-dthon-project/best.pt"

# 테스트 데이터 경로
# (!!중요!!: 이 경로들을 실제 테스트 데이터 경로로 수정해야 합니다.)
TEST_IMAGE_DIR = "/data/danielsohn0827000/unid/open/test/images" 
TEST_QUERY_DIR = "/data/danielsohn0827000/unid/open/test/query"   
SAMPLE_SUBMISSION_PATH = "/data/danielsohn0827000/unid/open/sample_submission.csv"
# 최종 제출 파일 이름
SUBMISSION_CSV_PATH = "submission.csv"

def build_query_to_image_map(query_dir, image_dir):
    """
    test/query 폴더의 모든 JSON을 파싱하여
    {query_id: image_path} 딕셔너리를 생성합니다.
    """
    print(f"Mapping test queries from {query_dir}...")
    query_map = {}
    query_json_files = glob.glob(os.path.join(query_dir, "*.json"))
    
    if not query_json_files:
        print(f"오류: {query_dir}에서 query json 파일을 찾을 수 없습니다.")
        return None

    for json_path in tqdm(query_json_files, desc="Building query-image map"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 1. JSON에서 이미지 파일명 획득
            img_name = data.get("source_data_info", {}).get("source_data_name_jpg")
            if not img_name:
                continue
            
            img_path = os.path.join(image_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found {img_path}")
                continue

            # 2. JSON 내부의 모든 쿼리(annotation)를 순회
            annotations = data.get("learning_data_info", {}).get("annotation", [])
            for query in annotations:
                # 3. 'instance_id' (query_id)와 'img_path'를 매핑
                query_id = query.get("instance_id")
                if query_id:
                    query_map[query_id] = img_path
                    
        except Exception as e:
            print(f"Warning: Failed to process {json_path}. Error: {e}")
            
    print(f"Mapped {len(query_map)} total queries to images.")
    return query_map

def main():
    
    # --- 2. 모델 로드 ---
    # model.py의 클래스를 인스턴스화. (모델 로딩은 여기서 한 번만)
    print("Loading QueryBasedDetector model...")
    model = QueryBasedDetector(best_pt_path=BEST_PT_PATH)
    print("Model loading complete.")

    # --- 3. 쿼리-이미지 맵 생성 ---
    query_map = build_query_to_image_map(TEST_QUERY_DIR, TEST_IMAGE_DIR)
    if query_map is None:
        return

    # --- 4. 추론 및 제출 파일 생성 ---
    
    # 샘플 제출 파일을 '작업 목록'으로 사용
    try:
        df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    except FileNotFoundError:
        print(f"오류: 샘플 제출 파일을 찾을 수 없습니다. 경로: {SAMPLE_SUBMISSION_PATH}")
        return
        
    predictions = [] # 예측된 bbox [x,y,w,h] 리스트

    print(f"Running inference on {len(df)} queries...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        query_id = row['query_id']
        query_text = row['query_text'] # .csv 파일에서 바로 쿼리 텍스트 사용
        
        img_path = query_map.get(query_id)
        
        if img_path:
            # model.py의 predict 함수 호출
            bbox_xywh = model.predict(img_path, query_text)
            predictions.append(bbox_xywh)
        else:
            # 맵에 없는 query_id (오류)
            print(f"Warning: Query ID {query_id} not found in map. Returning [0,0,0,0].")
            predictions.append([0.0, 0.0, 0.0, 0.0])

    # --- 5. 최종 CSV 저장 ---
    # 원본 DataFrame의 pred_ 컬럼들을 예측값으로 덮어쓰기
    pred_df = pd.DataFrame(predictions, columns=['pred_x', 'pred_y', 'pred_w', 'pred_h'])
    
    df['pred_x'] = pred_df['pred_x']
    df['pred_y'] = pred_df['pred_y']
    df['pred_w'] = pred_df['pred_w']
    df['pred_h'] = pred_df['pred_h']

    # 스크린샷과 동일한 형태로 저장 (query_text 컬럼 포함)
    df.to_csv(SUBMISSION_CSV_PATH, index=False)
    print(f"Inference complete. Submission file saved to: {SUBMISSION_CSV_PATH}")

if __name__ == "__main__":
    main()