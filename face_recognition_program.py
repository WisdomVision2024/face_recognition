import face_recognition as fr  # 用於人臉識別
import numpy as np  # 用於處理所有列表/陣列
import cv2  # 用於處理影像操作
import os  # 用於處理資料夾、路徑、圖片/檔案名稱等
import json  # 用於處理 JSON 格式
import concurrent.futures

# 已知人臉的資料夾路徑
faces_path = "C:\\DATA\\Topic\\face_images"

# 單張要識別的圖片路徑
image_to_recognize_path = "C:\\DATA\\Topic\\圖片測試\\57.jpg"  # 替換為您的圖片名稱

# 函數：獲取人臉名稱和人臉編碼
def get_face_encodings():
    face_names = os.listdir(faces_path)
    face_encodings = []

    # 使用迴圈檢索所有人臉編碼並存儲到列表中
    for i, name in enumerate(face_names):
        face = fr.load_image_file(f"{faces_path}\\{name}")
        face_encodings.append(fr.face_encodings(face)[0])
        face_names[i] = name.split(".")[0]  # 去除 ".jpg" 或其他圖片副檔名

    return face_encodings, face_names

# 加载人脸编码和名字
def load_cached_encodings():
    if os.path.exists("face_encodings.npy") and os.path.exists("face_names.npy"):
        face_encodings = np.load("face_encodings.npy", allow_pickle=True)
        face_names = np.load("face_names.npy", allow_pickle=True)
    else:
        face_encodings, face_names = get_face_encodings()
        np.save("face_encodings.npy", face_encodings)
        np.save("face_names.npy", face_names)
    return face_encodings, face_names

# 使用已經存在的函數來替代
face_encodings, face_names = load_cached_encodings()

# 加載要識別的圖片
image = fr.load_image_file(image_to_recognize_path)

# 獲取人臉位置座標和未知人臉編碼
face_locations = fr.face_locations(image)
unknown_encodings = fr.face_encodings(image, face_locations)

# 函數：處理每個人臉編碼
def process_face(face_encoding, face_location):
    face_distances = fr.face_distance(face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    best_distance = face_distances[best_match_index]
    confidence = max(0, 1 - best_distance)
    if confidence > 0.6:  # 設定匹配閾值
        name = face_names[best_match_index]
        top, right, bottom, left = face_location

        # 在人臉周圍畫矩形
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # 設置字體並顯示名字文字
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, f"{name} ({confidence:.2f})", (left, bottom + 20), font, 0.8, (255, 255, 255), 1)

        # 返回結果
        return {
            "name": name,
            "confidence": confidence
        }

# 用於存儲 JSON 結果的列表
results = []

# 使用多執行緒處理
with concurrent.futures.ThreadPoolExecutor() as executor:
    processed_results = executor.map(lambda p: process_face(*p), zip(unknown_encodings, face_locations))
    # 過濾 None 值
    results = [res for res in processed_results if res is not None]

# 將圖片從 RGB 轉換回 BGR，因為 OpenCV 使用 BGR
bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 在螢幕上顯示最終圖片
cv2.imshow("Image", bgr_image)

# 等待直到使用者按下任意鍵關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

# 輸出 JSON 格式的結果
json_result = json.dumps(results, indent=4)
print("識別結果 (JSON 格式):")
print(json_result)

