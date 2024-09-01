import face_recognition as fr  # 用於人臉識別
import numpy as np  # 用於處理所有列表/陣列
import cv2  # 用於處理影像操作
import os  # 用於處理資料夾、路徑、圖片/檔案名稱等
import json  # 用於處理 JSON 格式

# 已知人臉的資料夾路徑
faces_path = "C:\\DATA\\Topic\\face_images"

# 單張要識別的圖片路徑
image_to_recognize_path = "C:\\DATA\\Topic\\圖片測試\\10.jpg"  # 替換為您的圖片名稱


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


# 獲取人臉編碼並將其存儲到 face_encodings 變數中，還有名字
face_encodings, face_names = get_face_encodings()

# 加載要識別的圖片
image = fr.load_image_file(image_to_recognize_path)

# 獲取人臉位置座標和未知人臉編碼
face_locations = fr.face_locations(image)
unknown_encodings = fr.face_encodings(image, face_locations)

# 用於存儲 JSON 結果的列表
results = []

# 遍歷每個編碼及人臉位置
for face_encoding, face_location in zip(unknown_encodings, face_locations):
    # 計算與已知人臉編碼的距離
    face_distances = fr.face_distance(face_encodings, face_encoding)

    # 找到最小距離的索引（最接近的已知人臉編碼）
    best_match_index = np.argmin(face_distances)
    best_distance = face_distances[best_match_index]

    # 計算信心度（信心度可以設定為 1 - 距離）
    confidence = max(0, 1 - best_distance)  # 保證信心度不為負值

    # 設定匹配閾值（例如 0.6）
    if confidence > 0.6:  # 可根據需要調整閾值
        name = face_names[best_match_index]

        # 設定人臉位置的座標
        top, right, bottom, left = face_location

        # 在人臉周圍畫矩形
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # 設置字體並顯示名字文字
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, f"{name} ({confidence:.2f})", (left, bottom + 20), font, 0.8, (255, 255, 255), 1)

        # 將結果加入 JSON 列表
        results.append({
            "name": name,
            "confidence": confidence
        })

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
