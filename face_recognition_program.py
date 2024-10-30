import face_recognition as fr
import numpy as np
import cv2
import os
import json


class FaceRecognition:
    def __init__(self, faces_path):
        """
        初始化 FaceRecognition 類別
        :param faces_path: 已知人臉的資料夾路徑
        """
        self.faces_path = faces_path
        self.face_encodings, self.face_names = self.load_cached_encodings()

    def get_face_encodings(self):
        """
        獲取已知人臉的名稱和編碼
        :return: 人臉編碼列表和人臉名稱列表
        """
        face_names = os.listdir(self.faces_path)
        face_encodings = []

        for i, name in enumerate(face_names):
            face = fr.load_image_file(f"{self.faces_path}\\{name}")
            face_encodings.append(fr.face_encodings(face)[0])
            face_names[i] = name.split(".")[0]  # 去除圖片副檔名

        return face_encodings, face_names

    def load_cached_encodings(self):
        """
        加載已知人臉編碼，如果有緩存則從緩存載入
        :return: 人臉編碼和名稱列表
        """
        if os.path.exists("face_encodings.npy") and os.path.exists("face_names.npy"):
            face_encodings = np.load("face_encodings.npy", allow_pickle=True)
            face_names = np.load("face_names.npy", allow_pickle=True)
        else:
            face_encodings, face_names = self.get_face_encodings()
            np.save("face_encodings.npy", face_encodings)
            np.save("face_names.npy", face_names)
        return face_encodings, face_names

    def recognize_faces(self, image_path):
        """
        對指定圖片進行人臉識別並返回結果
        :param image_path: 要識別的圖片路徑
        :return: 識別結果（包含名字和信心度）以及標註人臉的圖片
        """
        # 加載要識別的圖片
        image = fr.load_image_file(image_path)

        # 獲取人臉位置座標和未知人臉編碼
        face_locations = fr.face_locations(image)
        unknown_encodings = fr.face_encodings(image, face_locations)

        results = []  # 存儲識別結果的列表

        # 遍歷每個人臉編碼及位置
        for face_encoding, face_location in zip(unknown_encodings, face_locations):
            # 計算與已知人臉的距離
            face_distances = fr.face_distance(self.face_encodings, face_encoding)

            # 找到最接近的已知人臉
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            # 計算信心度
            confidence = max(0, 1 - best_distance)

            if confidence > 0.6:  # 設定匹配閾值
                name = self.face_names[best_match_index]
                top, right, bottom, left = face_location

                # 在人臉周圍畫矩形
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

                # 顯示名字和信心度
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, f"{name} ({confidence:.2f})", (left, bottom + 20), font, 0.8, (255, 255, 255), 1)

                results.append({
                    "name": name,
                    "confidence": confidence
                })

        # 將圖片從 RGB 轉換為 BGR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return results, bgr_image

    def get_results_json(self, image_path):
        """
        獲取識別結果的 JSON 格式
        :param image_path: 要識別的圖片路徑
        :return: JSON 格式的識別結果
        """
        results, _ = self.recognize_faces(image_path)
        return json.dumps(results, indent=4)


# 使用範例
if __name__ == "__main__":
    # 初始化 FaceRecognition 類別
    face_recognition_system = FaceRecognition("C:\\DATA\\Topic\\face_images")

    # 識別圖片中的人臉
    results, annotated_image = face_recognition_system.recognize_faces("C:\\DATA\\Topic\\圖片測試\\57.jpg")

    # 顯示結果圖片
    cv2.imshow("Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 輸出 JSON 格式的結果
    json_result = face_recognition_system.get_results_json("C:\\DATA\\Topic\\圖片測試\\57.jpg")
    print("識別結果 (JSON 格式):")
    print(json_result)
