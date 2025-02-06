import cv2
import os
import numpy as np

CASCADE_PATH = "cascades/"
MODEL_PATH = "models/"
OUTPUT_PATH = "output/"

face_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, "haarcascade_frontalface_default.xml"))
eye_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, "haarcascade_eye.xml"))

def load_yolo_model():
    net = cv2.dnn.readNetFromDarknet(os.path.join(MODEL_PATH, "yolov3-face.cfg"),
                                     os.path.join(MODEL_PATH, "yolov3-wider_16000.weights"))
    return net

def detect_with_yolo(frame, net, conf_threshold=0.5, nms_threshold=0.4):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    h, w = frame.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Face: {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

def detect_with_haar(frame):
    """
    face detection using Haar cascades
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frame

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    while True:
        print("Выберите тип обнаружения:")
        print("1. Лицо с использованием каскадов Хаара")
        print("2. Лицо с использованием YOLO")
        print("3. Завершить работу")
        option = input("Введите номер опции: ")

        if option == '3':
            print("Программа завершена.")
            break

        if option == '1':
            detection_method = "haar"
        elif option == '2':
            detection_method = "yolo"
            yolo_net = load_yolo_model()
        else:
            print("Некорректный ввод, попробуйте снова.")
            continue

        print("Выберите источник:")
        print("1. Камера")
        print("2. Видео файл")
        print("3. Изображение")
        source_option = input("Введите номер источника: ")

        if source_option == '1':
            cap = cv2.VideoCapture(0)
            output_video_path = os.path.join(OUTPUT_PATH, "camera_result.mp4")
        elif source_option == '2':
            video_file = input("Введите имя видеофайла (например, 1.mp4): ")
            cap = cv2.VideoCapture(os.path.join("data", video_file))
            if not cap.isOpened():
                print("Ошибка при открытии видеофайла.")
                continue
            output_video_path = os.path.join(OUTPUT_PATH, f"{os.path.splitext(video_file)[0]}_result.mp4")
        elif source_option == '3':
            image_file = input("Введите имя изображения (например, 1.jpg): ")
            image_path = os.path.join("data", image_file)
            if os.path.exists(image_path):
                frame = cv2.imread(image_path)
                if detection_method == "haar":
                    frame = detect_with_haar(frame)
                elif detection_method == "yolo":
                    frame = detect_with_yolo(frame, yolo_net)
                output_image_path = os.path.join(OUTPUT_PATH, f"{os.path.splitext(image_file)[0]}_result.jpg")
                cv2.imwrite(output_image_path, frame)
                print(f"Результат сохранен: {output_image_path}")
                cv2.imshow("Result", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Файл изображения не найден.")
            continue
        else:
            print("Некорректный ввод, попробуйте снова.")
            continue

        if cap.isOpened():
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if detection_method == "haar":
                    frame = detect_with_haar(frame)
                elif detection_method == "yolo":
                    frame = detect_with_yolo(frame, yolo_net)

                out.write(frame)
                cv2.imshow("Video", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            print(f"Результат видео сохранен: {output_video_path}")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
