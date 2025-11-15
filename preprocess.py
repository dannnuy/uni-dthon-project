# 파일명: preprocess.py
# (수정된 버전)

import os
import json
import glob
import shutil
from tqdm import tqdm

# --- 사용자 설정 ---
# 해커톤 데이터셋의 'train_valid' 폴더 경로
INPUT_DATA_DIR = "/data/danielsohn0827000/unid/train_valid" 
# YOLO 데이터셋이 생성될 경로
OUTPUT_DATA_DIR = "/data/danielsohn0827000/uni/yolo_dataset"

# data.yaml 파일을 생성할 때 사용할 클래스 맵 (출력용)
# 0번: 표, 1번: 차트
YAML_CLASS_MAP = {
    0: "표",
    1: "차트"
}
# --------------------

def convert_to_yolo_format(bbox, img_w, img_h):
    """
    JSON의 [x_min, y_min, width, height] 포맷을
    YOLO의 [x_center_norm, y_center_norm, width_norm, height_norm] 포맷으로 변환합니다.
    """
    x_min, y_min, w, h = bbox
    
    if w <= 0 or h <= 0:
        return None
    
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    
    x_center_norm = min(1.0, x_center / img_w)
    y_center_norm = min(1.0, y_center / img_h)
    w_norm = min(1.0, w / img_w)
    h_norm = min(1.0, h / img_h)
    
    return x_center_norm, y_center_norm, w_norm, h_norm

def process_dataset(subset):
    """
    주어진 subset('train' 또는 'valid')에 대해 전처리 작업을 수행합니다.
    """
    print(f"\nProcessing '{subset}' subset...")
    
    json_dirs = [
        os.path.join(INPUT_DATA_DIR, subset, "press_json"),
        os.path.join(INPUT_DATA_DIR, subset, "report_json")
    ]
    img_dirs = {
        "press_json": os.path.join(INPUT_DATA_DIR, subset, "press_jpg"),
        "report_json": os.path.join(INPUT_DATA_DIR, subset, "report_jpg")
    }
    
    img_output_dir = os.path.join(OUTPUT_DATA_DIR, "images", subset)
    label_output_dir = os.path.join(OUTPUT_DATA_DIR, "labels", subset)
    
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    
    json_files = []
    for d in json_dirs:
        json_files.extend(glob.glob(os.path.join(d, "*.json")))
        
    for json_path in tqdm(json_files, desc=f"Converting {subset} data"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            img_info = data.get("source_data_info")
            if not img_info:
                print(f"Warning: Skipping {json_path}. 'source_data_info' key missing.")
                continue
                
            resolution = img_info.get("document_resolution")
            if not resolution or not isinstance(resolution, list) or len(resolution) != 2:
                print(f"Warning: Skipping {json_path}. Invalid or missing 'document_resolution'.")
                continue
            
            img_w, img_h = resolution
            if img_w <= 0 or img_h <= 0:
                print(f"Warning: Skipping {json_path}. Invalid image dimensions (w={img_w}, h={img_h}).")
                continue

            jpg_name_from_json = img_info.get("source_data_name_jpg")
            if not jpg_name_from_json:
                print(f"Warning: Skipping {json_path}. 'source_data_name_jpg' key missing.")
                continue

            annotations = data.get("learning_data_info", {}).get("annotation", [])
            if not annotations:
                continue
            
            yolo_labels = []
            
            for ann in annotations:
                class_name = ann.get("class_name")
                class_id = None  # 기본값 초기화

                # --- ✅ 여기가 핵심 수정 사항 ---
                if class_name == "표":
                    class_id = 0
                elif class_name and class_name.startswith("차트"): # "차트(...)"로 시작하는 모든 경우
                    class_id = 1
                # -----------------------------

                # class_id가 0 또는 1로 할당된 경우 (표 또는 차트인 경우)
                if class_id is not None:
                    bbox = ann.get("bounding_box") # [x_min, y_min, width, height]
                    
                    if (bbox and isinstance(bbox, list) and len(bbox) == 4):
                        yolo_result = convert_to_yolo_format(bbox, img_w, img_h)
                        
                        if yolo_result:
                            x_c_n, y_c_n, w_n, h_n = yolo_result
                            yolo_labels.append(f"{class_id} {x_c_n} {y_c_n} {w_n} {h_n}")
                        else:
                            print(f"Warning: Skipping invalid bbox (w/h <= 0) in {json_path}")
                    else:
                        print(f"Warning: Skipping annotation in {json_path}. Invalid bbox format: {bbox}")

            base_filename = os.path.splitext(os.path.basename(json_path))[0]
            
            if yolo_labels: # 유효한 라벨(표 또는 차트)이 하나라도 있을 경우에만
                label_path = os.path.join(label_output_dir, f"{base_filename}.txt")
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_labels))
                    
                src_img_path = None
                json_dir_name = os.path.basename(os.path.dirname(json_path))
                
                if json_dir_name in img_dirs:
                    potential_path = os.path.join(img_dirs[json_dir_name], jpg_name_from_json)
                   
                    if os.path.exists(potential_path):
                         src_img_path = potential_path
                    else:
                        fallback_jpg_name = f"{base_filename}.jpg"
                        potential_path_fb = os.path.join(img_dirs[json_dir_name], fallback_jpg_name)
                        if os.path.exists(potential_path_fb):
                            src_img_path = potential_path_fb
                        else:
                            print(f"Warning: Could not find image {jpg_name_from_json} (or fallback) for {json_path}")
                
                if src_img_path:
                    dst_img_path = os.path.join(img_output_dir, f"{base_filename}.jpg")
                    shutil.copy(src_img_path, dst_img_path)

        except json.JSONDecodeError:
            print(f"Error: Skipping {json_path}. File is not a valid JSON.")
        except Exception as e:
            print(f"Error processing {json_path}: {e}. Skipping this file.")

def create_yaml_file():
    """
    YOLO 훈련을 위한 data.yaml 파일을 생성합니다.
    """
    # YAML_CLASS_MAP의 key(0, 1)를 기준으로 정렬
    sorted_keys = sorted(YAML_CLASS_MAP.keys())
    class_names = [YAML_CLASS_MAP[key] for key in sorted_keys]
    
    yaml_content = f"""
train: {os.path.abspath(os.path.join(OUTPUT_DATA_DIR, 'images/train'))}
val: {os.path.abspath(os.path.join(OUTPUT_DATA_DIR, 'images/val'))}

# number of classes
nc: {len(class_names)}

# class names
names: {class_names}
"""
    
    yaml_path = os.path.join(OUTPUT_DATA_DIR, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"\nSuccessfully created 'data.yaml' at {yaml_path}")
    print(f"Class names: {class_names}")

def main():
    if os.path.exists(OUTPUT_DATA_DIR):
        print(f"Output directory '{OUTPUT_DATA_DIR}' already exists. Removing it.")
        shutil.rmtree(OUTPUT_DATA_DIR)
        
    process_dataset("train")
    process_dataset("valid")
    create_yaml_file()
    print("\nPreprocessing complete. YOLO dataset is ready.")

if __name__ == "__main__":
    main()