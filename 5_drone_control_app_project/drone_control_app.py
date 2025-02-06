import cv2
import json
import os
import threading
import tkinter as tk
from clover import srv
from std_srvs.srv import Trigger
from tkinter import filedialog, messagebox
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import Range
import csv
from clover.srv import SetLEDEffect
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from PIL import Image as PILImage, ImageTk
import numpy as np
import math
import time

# Инициализация ROS-ноды
rospy.init_node('flight_control_gui')
bridge = CvBridge()

# Загрузка каскада ХААР и YOLO-моделей
try:
    fullbody_cascade = cv2.CascadeClassifier('cascades/haarcascade_fullbody.xml')
    background_sub = cv2.createBackgroundSubtractorMOG2()
    net = cv2.dnn.readNet("models/yolov3-face.cfg", "models/yolov3-wider_16000.weights")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except Exception as e:
    print(f"Ошибка загрузки моделей: {e}")
    exit(1)

# Инициализация переменных
flight_plan = None
video_writer = None
latest_image = None
lock = threading.Lock()
running_mode = None
current_range = None
telemetry_recording = False

# ROS-сервисы
try:
    get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)
    navigate = rospy.ServiceProxy("navigate", srv.Navigate)
    navigate_global = rospy.ServiceProxy("navigate_global", srv.NavigateGlobal)
    land = rospy.ServiceProxy("land", Trigger)
    set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect, persistent=True)
except rospy.ServiceException as e:
    print(f"Ошибка при инициализации служб ROS: {e}")
    exit(1)

# Установка домашней позиции
try:
    start = get_telemetry()
    home_position = [start.lat, start.lon]
    flag = True
except Exception as e:
    print(f"Ошибка получения телеметрии: {e}")
    exit(1)

# Открытие csv документа
def load_data():
    file_path = filedialog.askopenfilename(title="Выберите файл",
                                           filetypes=(("CSV файлы", "*.csv"), ("Все файлы", "*.*")))

    if not file_path:
        print("Файл не выбран.")
        return [], []

    times = []
    heights = []

    try:
        with open(file_path, newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            header = next(reader, None)

            if header is None:
                print("Файл пустой или имеет неверный формат.")
                return [], []

            for row in reader:
                if len(row) < 4:
                    print(f"Пропущена строка с недостаточным количеством данных: {row}")
                    continue

                try:
                    time = datetime.strptime(row[0], "%Y-%m-%d_%H-%M-%S")
                    height = float(row[3])
                    times.append(time)
                    heights.append(height)
                except (ValueError, IndexError) as e:
                    print(f"Ошибка обработки строки {row}: {e}")
                    continue
    except Exception as e:
        print(f"Ошибка открытия файла {file_path}: {e}")
        return [], []

    return times, heights

# Построение графика
def plot_graph():
    times, heights = load_data()
    if not times or not heights:
        status_label.config(text="Ошибка: данные не загружены или файл пуст.")
        return

    if len(times) != len(heights):
        status_label.config(text="Ошибка: несоответствие данных времени и высоты.")
        return

    # Очистка старых графиков, если есть
    for widget in window.winfo_children():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().destroy()

    # Создание графика
    fig, ax = plt.subplots(figsize=(6, 4))

    # Вычисление времени в секундах
    start_time = times[0]
    times_in_seconds = [(t - start_time).total_seconds() for t in times]

    # Построение графика
    ax.plot(times_in_seconds, heights, label="График высоты", color='blue', marker='o', linestyle='-')
    ax.set_xlabel("Время (секунды)")
    ax.set_ylabel("Высота (м)")
    ax.set_title("Изменение высоты с течением времени")
    ax.legend()

    # Встраивание графика в интерфейс
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().place(x=800, y=50)

# Получение данных о расстоянии
def range_callback(msg):
    """
    Callback для подписки на данные
    """
    global current_range
    try:
        # Проверка на корректность значения и игнорируем отрицательные значения
        if msg.range >= 0:
            current_range = msg.range
        else:
            rospy.logwarn("Получено некорректное значение расстояния")
    except Exception as e:
        rospy.logerr(f"Ошибка обработки данных дальномера: {e}")

# Подписка на топик
def subscribe_to_rangefinder():
    rospy.Subscriber('rangefinder/range', Range, range_callback)

# Функция для обновления данных на интерфейсе
def update_range_label():
    if current_range is not None:
        range_label.config(text=f"Расстояние: {current_range:.2f} м")
    else:
        range_label.config(text="Расстояние: Неизвестно")

    # Повторный вызов через 1 секунду
    window.after(1000, update_range_label)

# Инициализация подписки
subscribe_to_rangefinder()

# Включение светодиодов
def turn_on_led():
    try:
        # Установка светодиодного эффекта
        response = set_effect(effect='rainbow_fill')
        if response.success:
            status_label.config(text="Светодиод включен", fg="green")
        else:
            status_label.config(text="Не удалось включить светодиод", fg="red")
            rospy.logwarn("Сервис set_effect вернул ошибку")
    except rospy.ServiceException as e:
        # Обработка ошибок вызова сервиса
        status_label.config(text="Ошибка вызова сервиса светодиода", fg="red")
        rospy.logerr(f"Ошибка при включении светодиода: {e}")
    except Exception as e:
        # Обработка любых других исключений
        status_label.config(text="Неизвестная ошибка при включении светодиода", fg="red")
        rospy.logerr(f"Неизвестная ошибка при включении светодиода: {e}")

# Создание файла csv
def create_telemetry_csv():
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"telemetry_{time}.csv"

    try:
        with open(filename, mode="w", newline='') as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(['Time', 'X', 'Y', 'Z', 'Lat', 'Lon', 'Speed'])
        return filename
    except Exception as e:
        rospy.logerr(f"Ошибка создания файла CSV: {e}")
        return None

# Запись телеметрии
def record_telemetry_to_csv(filename):
    try:
        telem = get_telemetry()
        with open(filename, mode="a", newline='') as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(
                [datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                 telem.x,
                 telem.y,
                 telem.z,
                 telem.lat,
                 telem.lon,
                 math.sqrt(telem.vx ** 2 + telem.vy ** 2 + telem.vz ** 2)]
            )
    except Exception as e:
        rospy.logerr(f"Ошибка записи телеметрии в файл {filename}: {e}")
        status_label.config(text=f"Ошибка записи телеметрии: {e}", fg="red")

# Функция начала записи телеметрии
def start_telemetry_recording():
    global telemetry_recording, telemetry_filename

    if telemetry_recording:
        telemetry_recording = False
        csv_telem_button.config(text="Запись телеметрии", bg="blue")
        rospy.loginfo("Запись телеметрии остановлена.")
        return

    telemetry_filename = create_telemetry_csv()
    if telemetry_filename is None:
        status_label.config(text="Ошибка создания файла для телеметрии", fg="red")
        return

    telemetry_recording = True
    csv_telem_button.config(text="Остановить запись", bg="red")
    rospy.loginfo(f"Начата запись телеметрии в файл {telemetry_filename}.")

    def record_telemetry():
        while telemetry_recording:
            record_telemetry_to_csv(telemetry_filename)
            rospy.sleep(5)

    threading.Thread(target=record_telemetry, daemon=True).start()

# Обновление высоты
def update_altitude():
    try:
        telem = get_telemetry()
        altitude = telem.z
        alt_label.config(text=f"Текущая высота: {altitude:.2f} м", fg="blue")
    except rospy.ServiceException as e:
        alt_label.config(text="Ошибка получения высоты", fg="red")
        rospy.logerr(f"Ошибка вызова службы телеметрии: {e}")
    except Exception as e:
        alt_label.config(text="Неизвестная ошибка получения высоты", fg="red")
        rospy.logerr(f"Неизвестная ошибка в update_altitude: {e}")

    window.after(1000, update_altitude)

# Ожидание прибытия дрона
def arrival_wait(tolerance=0.2):
    while not rospy.is_shutdown():
        try:
            telem = get_telemetry(frame_id="navigate_target")
            distance = math.sqrt(telem.x**2 + telem.y**2 + telem.z**2)
            if distance < tolerance:
                rospy.loginfo("Цель достигнута.")
                break
        except rospy.ServiceException as e:
            rospy.logerr(f"Ошибка получения телеметрии для ожидания прибытия: {e}")
        except Exception as e:
            rospy.logerr(f"Неизвестная ошибка в arrival_wait: {e}")
        rospy.sleep(0.2)

# Взлет
def takeoff():
    try:
        # Получение высоты и скорости из пользователя
        z = float(entry_z.get()) if entry_z.get() else 3.0  # Значение по умолчанию 3 м
        speed = float(entry_speed.get()) if entry_speed.get() else 1.0  # Значение по умолчанию 1 м/с

        # Вызов навигации в отдельном потоке
        threading.Thread(target=navigate, args=(0, 0, z, speed, 1, "body", True)).start()
        status_label.config(text=f"Взлет на высоту {z:.2f} м со скоростью {speed:.2f} м/с", fg="blue")

    except ValueError:
        status_label.config(text="Ошибка: введите корректные числовые значения для высоты и скорости", fg="red")
    except Exception as e:
        status_label.config(text=f"Неизвестная ошибка при взлете: {e}", fg="red")

# Посадка
def land_drone():
    try:
        threading.Thread(target=land).start()
        status_label.config(text="Дрон приземляется...", fg="blue")
    except Exception as e:
        status_label.config(text=f"Ошибка при посадке: {e}", fg="red")

# Полет по локальным координатам
def fly_to_local_coordinates():
    try:
        # Получение координат из полей ввода от пользователя
        x = float(entry_x.get())
        y = float(entry_y.get())
        z = float(entry_z.get()) if entry_z.get() else 0.0  # Значение по умолчанию 0 м
        speed = float(entry_speed.get()) if entry_speed.get() else 1.0  # Значение по умолчанию 1 м/с

        # Вызов навигации в отдельном потоке
        threading.Thread(target=navigate, args=(x, y, z, speed, 1, "body", True)).start()
        status_label.config(text=f"Полет: X={x:.2f}, Y={y:.2f}, Z={z:.2f}, скорость={speed:.2f} м/с", fg="blue")

    except ValueError:
        status_label.config(text="Ошибка: введите корректные числовые значения для X, Y, высоты и скорости", fg="red")
    except Exception as e:
        status_label.config(text=f"Неизвестная ошибка при полете по локальным координатам: {e}", fg="red")

# Полет по глобальным координатам
def fly_to_global_coordinates():
    try:
        # Получение координат из полей ввода от пользователя
        lat = float(entry_lat.get())
        lon = float(entry_lon.get())
        z = float(entry_z.get()) if entry_z.get() else 3.0  # Значение по умолчанию 3 м
        speed = float(entry_speed.get()) if entry_speed.get() else 1.0  # Значение по умолчанию 1 м/с

        # Вызов глобальной навигации в отдельном потоке
        threading.Thread(target=navigate_global, args=(lat, lon, z, speed, 1, "map", False)).start()
        status_label.config(text=f"Полет к широте: {lat:.6f}, долготе: {lon:.6f}, высота: {z:.2f} м", fg="blue")

    except ValueError:
        status_label.config(text="Ошибка: введите корректные числовые значения для широты, долготы, высоты и скорости", fg="red")
    except Exception as e:
        status_label.config(text=f"Неизвестная ошибка при полете по глобальным координатам: {e}", fg="red")

# Возврат на начальную точку
def fly_home():
    try:
        if home_position:
            # Получение высоты и скорости из полей ввода
            z = float(entry_z.get()) if entry_z.get() else 3.0  # Значение по умолчанию 3 м
            speed = float(entry_speed.get()) if entry_speed.get() else 1.0  # Значение по умолчанию 1 м/с

            lat, lon = home_position[0], home_position[1]

            # Вызов глобальной навигации для возврата домой
            threading.Thread(target=navigate_global, args=(lat, lon, z, speed, 1, "map", False)).start()
            status_label.config(text="Возвращение домой...", fg="blue")

            # Ожидание прибытия и посадка
            arrival_wait()
            land_drone()
        else:
            status_label.config(text="Ошибка: точка взлета (дом) не определена", fg="red")

    except ValueError:
        status_label.config(text="Ошибка: введите корректные числовые значения для высоты и скорости", fg="red")
    except Exception as e:
        status_label.config(text=f"Неизвестная ошибка при возврате домой: {e}", fg="red")

# Показ телеметрии
def show_telemetry():
    try:
        telem = get_telemetry()
        status_label.config(text=f"X = {telem.x:.2f}, Y = {telem.y:.2f}, Z = {telem.z:.2f}")
    except Exception as e:
        status_label.config(text=f"Ошибка получения телеметрии: {e}", fg="red")

# Загрузка плана
def load_plan_file(filename):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        messagebox.showerror("Ошибка загрузки", "Не удалось декодировать файл. Возможно, файл поврежден или имеет неправильный формат.")
        return None
    except FileNotFoundError:
        messagebox.showerror("Ошибка загрузки", "Файл не найден.")
        return None
    except Exception as e:
        messagebox.showerror("Ошибка загрузки", f"Неизвестная ошибка при загрузке файла: {e}")
        return None

def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("Flight Plan Files", "*.plan")])
    if filename:
        if filename.endswith(".plan"):
            flight_plan = load_plan_file(filename)
            if flight_plan:
                status_label.config(text=f"Файл плана полета {filename} успешно загружен", fg="blue")
            else:
                status_label.config(text="Ошибка загрузки плана полета", fg="red")
        else:
            messagebox.showerror("Неверный файл", "Выберите файл с расширением .plan")

# Полет по плану
def fly_by_plan():
    if flight_plan is None:
        messagebox.showerror("Ошибка", "План полета не загружен")
        return

    def run_flight_plan():
        try:
            home_lat = flight_plan["mission"]["plannedHomePosition"][0]
            home_lon = flight_plan["mission"]["plannedHomePosition"][1]

            # Проходим по всем пунктам плана полета
            for item in flight_plan["mission"]["items"]:
                command = item.get("command")

                if command == 22:  # Команда взлета
                    if get_telemetry().armed:
                        # Возвращаемся домой и приземляемся
                        navigate_global(lat=home_lat, lon=home_lon, z=3, yaw=math.inf, speed=1, frame_id='map')
                        arrival_wait()
                        land()
                    else:
                        z = item["params"][6]  # Извлекаем высоту для команды взлета
                        navigate_global(lat=home_lat, lon=home_lon, z=z, yaw=math.inf, speed=1, frame_id='map',
                                        auto_arm=True)
                        navigate_global(lat=home_lat, lon=home_lon, z=z, yaw=math.inf, speed=1, frame_id='map')
                        arrival_wait()

                elif command == 16:  # Команда перемещения на координаты
                    lat = item["params"][4]
                    lon = item["params"][5]
                    z = item["params"][6] if len(item["params"]) > 6 else 3  # Используем высоту из плана, если она есть
                    navigate_global(lat=lat, lon=lon, z=z, yaw=math.inf, speed=1, frame_id='map')
                    arrival_wait()

                elif command == 20:  # Команда возврата домой
                    navigate_global(lat=home_lat, lon=home_lon, z=3, yaw=math.inf, speed=1, frame_id='map')
                    arrival_wait()

                elif command == 21:  # Команда приземления
                    land()

            status_label.config(text="План полета завершен")

        except Exception as e:
            # Если возникла ошибка во время выполнения плана полета
            status_label.config(text=f"Ошибка выполнения плана: {e}", fg="red")

    threading.Thread(target=run_flight_plan, daemon=True).start()

# Обработка и вывод изображения с камеры
def camera_image(msg):
    global latest_image
    with lock:
        try:
            latest_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            print(f"Ошибка при конвертации изображения: {e}")

image_sub = rospy.Subscriber('main_camera/image_raw', Image, camera_image, queue_size=1)

# Обновление изображения в Tkinter
def update_image():
    if latest_image is not None:
        try:
            with lock:
                img_rgb = cv2.cvtColor(latest_image, cv2.COLOR_BGR2RGB)
                img_pil = PILImage.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                camera_label.config(image=img_tk)
                camera_label.image = img_tk
        except Exception as e:
            print(f"Ошибка обновления изображения: {e}")
    window.after(100, update_image)  # Обновление изображения через 100ms

# Распознание объекта с каскадом
def detected_objects():
    global latest_image
    while running_mode == "objects":
        if latest_image is None:
            time.sleep(0.1)  # Если нет нового изображения, нужно подождать немного перед повторной проверкой
            continue
        try:
            with lock:
                img = latest_image.copy()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            objects = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))

            for (x, y, w, h) in objects:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            display_image(img)
        except Exception as e:
            print(f"Ошибка распознавания объектов: {e}")
            continue

# Запуск режима распознания объектов с каскадом
def start_object_detection():
    stop_detection()
    global running_mode
    running_mode = "objects"
    threading.Thread(target=detected_objects, daemon=True).start()

# Остановка распознания
def stop_detection():
    global running_mode
    running_mode = None

# Распознание движущихся объектов
def detect_motion():
    global latest_image
    while running_mode == "motion":
        if latest_image is None:
            time.sleep(0.1)  # Ждем..., если изображения нет
            continue
        try:
            with lock:
                img = latest_image.copy()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = background_sub.apply(gray)
            _, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 400:
                    cv2.polylines(img, [contour], isClosed=True, color=(0, 0, 255), thickness=1)

            display_image(img)
        except Exception as e:
            print(f"Ошибка при детекции движения: {e}")

# Распознание объектов по цвету
def detect_by_color():
    global latest_image
    while running_mode == "color_detection":
        if latest_image is None:
            time.sleep(0.1)  # Ждем..., если изображения нет
            continue
        try:
            with lock:
                img = latest_image.copy()

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 100, 50])
            upper_green = np.array([85, 255, 255])
            lower_blue = np.array([100, 150, 50])
            upper_blue = np.array([140, 255, 255])

            green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
            blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in green_contours:
                if cv2.contourArea(contour) > 200:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for contour in blue_contours:
                if cv2.contourArea(contour) > 200:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            display_image(img)
        except Exception as e:
            print(f"Ошибка при детекции по цвету: {e}")
       
# Запуск режима распознания по цветам
def start_detect_color():
    stop_detection()
    global running_mode
    running_mode = "color_detection"
    print("start color detection")
    threading.Thread(target=detect_by_color, daemon=True).start()

# Запуск режима распознавания движущихся объектов
def start_motion_detection():
    stop_detection()
    global running_mode
    running_mode = "motion"
    print("start motion detection")
    threading.Thread(target=detect_motion, daemon=True).start()

# Распознавание лиц с помощью YOLO
def detect_faces_with_yolo():
    global latest_image
    while running_mode == "face_detection":
        if latest_image is None:
            time.sleep(0.1)  # Даем немного времени..., если изображения нет
            continue

        try:
            with lock:
                img = latest_image.copy()

            # Преобразование изображения в формат, который понимает YOLO
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Обработка результатов YOLO
            class_ids = []
            confidences = []
            boxes = []
            height, width, channels = img.shape

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Коэффициент уверенности ```💪```
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Применение NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Отображение результатов на изображении
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            display_image(img)

        except Exception as e:
            print(f"Ошибка при детекции лиц с YOLO: {e}")

# Запуск распознавания лиц с использованием YOLO
def start_face_detection():
    stop_detection()
    global running_mode
    running_mode = "face_detection"
    print("start face detection")
    threading.Thread(target=detect_faces_with_yolo, daemon=True).start()

# Отображение видео
def display_image(img):
    global video_writer
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = PILImage.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    camera_label.config(image=img_tk)
    camera_label.image = img_tk

    # Запись видео, если writer существует
    if video_writer is not None:
        video_writer.write(img)

# Старт записи видео
def start_video_recording():
    global video_writer
    if video_writer is not None:
        messagebox.showinfo("Запись", "Видео уже записывается")
        return

    # Создание папки для записи видео, если она не существует
    output_dir = 'video_output'
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать папку для видео: {e}")
            return

    # Имя файла с текущей датой и временем
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(output_dir, f"output_{time_str}.avi")

    # Создание объекта для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (320, 240))

    # Функция записи видео
    def record():
        global latest_image
        while video_writer is not None:
            with lock:  # Синхронизация доступа к latest_image
                if latest_image is None:
                    time.sleep(0.03)  # Охлаждение потока, чтобы избежать высокой загрузки
                    continue
                video_writer.write(latest_image)
            time.sleep(0.03)  # Пауза между кадрами

    # Запуск записи видео в отдельном потоке
    threading.Thread(target=record, daemon=True).start()

# Остановка записи видео
def stop_video_recording():
    global video_writer
    if video_writer is not None:
        try:
            video_writer.release()
            video_writer = None
            messagebox.showinfo("Запись", "Видео успешно сохранено")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении видео: {e}")
    else:
        messagebox.showinfo("Запись", "Запись не была начата")

# Окно App
window = tk.Tk()
window.title("Управление дроном")
window.geometry("1080x650")

# Поля ввода и лейблы
tk.Label(window, text="Высота взлета (м): ").grid(row=0, column=0, padx=20, pady=10, sticky="w")
entry_z = tk.Entry(window, width=15)
entry_z.grid(row=0, column=1, padx=20, pady=10)

tk.Label(window, text="Координата X (м): ").grid(row=1, column=0, padx=20, pady=10, sticky="w")
entry_x = tk.Entry(window, width=15)
entry_x.grid(row=1, column=1, padx=20, pady=10)

tk.Label(window, text="Координата Y (м): ").grid(row=2, column=0, padx=20, pady=10, sticky="w")
entry_y = tk.Entry(window, width=15)
entry_y.grid(row=2, column=1, padx=20, pady=10)

tk.Label(window, text="Широта: ").grid(row=3, column=0, padx=20, pady=10, sticky="w")
entry_lat = tk.Entry(window, width=15)
entry_lat.grid(row=3, column=1, padx=20, pady=10)

tk.Label(window, text="Долгота: ").grid(row=4, column=0, padx=20, pady=10, sticky="w")
entry_lon = tk.Entry(window, width=15)
entry_lon.grid(row=4, column=1, padx=20, pady=10)

tk.Label(window, text="Скорость (м/с): ").grid(row=5, column=0, padx=20, pady=10, sticky="w")
entry_speed = tk.Entry(window, width=15)
entry_speed.grid(row=5, column=1, padx=20, pady=10)

# Кнопки управления
takeoff_button = tk.Button(window, text="Взлет", width=20, bg="brown3", fg="white", relief="solid", command=takeoff)
takeoff_button.grid(row=6, column=0, padx=20, pady=5)

land_button = tk.Button(window, text="Посадка", width=20, relief="solid", command=land)
land_button.grid(row=7, column=0, padx=20, pady=5)

global_coordinates_button = tk.Button(window, text="Гл. Координаты", width=20, relief="solid", command=fly_to_global_coordinates)
global_coordinates_button.grid(row=8, column=0, padx=20, pady=5)

load_plan_button = tk.Button(window, text="Загрузить план", width=20, relief="solid", command=browse_file)
load_plan_button.grid(row=10, column=0, padx=20, pady=5)

home_button = tk.Button(window, text="Домой", width=20, relief="solid", command=fly_home)
home_button.grid(row=6, column=1, padx=20, pady=5)

telemetry_button = tk.Button(window, text="Телеметрия", width=20, relief="solid", command=show_telemetry)
telemetry_button.grid(row=7, column=1, padx=20, pady=5)

local_coordinates_button = tk.Button(window, text="Лок. Координаты", width=20, relief="solid", command=fly_to_local_coordinates)
local_coordinates_button.grid(row=8, column=1, padx=20, pady=5)

activate_plan_button = tk.Button(window, text="Активировать план", width=20, relief="solid", command=fly_by_plan)
activate_plan_button.grid(row=10, column=1, padx=20, pady=5)

status_label = tk.Label(window, text="Состояние дрона", fg="blue")
status_label.grid(row=11, column=0, columnspan=2)

alt_label = tk.Label(window, text="Текущая высота", fg="blue")
alt_label.grid(row=11, column=3, columnspan=2)

range_label = tk.Label(window, text="Текущее расстояние", fg="green")
range_label.grid(row=12, column=3, columnspan=2)

# Кнопки для камеры
detection_button = tk.Button(window, text="Распознать объект", width=20, bg="blue", fg="white", relief="solid", command=start_object_detection)
detection_button.grid(row=7, column=3, padx=20, pady=5)

detection_move_button = tk.Button(window, text="Распознать движение", width=20, bg="blue", fg="white", relief="solid", command=start_motion_detection)
detection_move_button.grid(row=8, column=3, padx=20, pady=5)

detection_color_button = tk.Button(window, text="Распознать цвета", width=20, bg="blue", fg="white", relief="solid", command=start_detect_color)
detection_color_button.grid(row=9, column=3, padx=20, pady=5)

face_detection_button = tk.Button(window, text="Распознать лицо", width=20, bg="blue", fg="white", relief="solid", command=start_face_detection)
face_detection_button.grid(row=9, column=4, padx=20, pady=5)

stop_detection_button = tk.Button(window, text="Остановить", width=20, bg="red", fg="white", relief="solid", command=stop_detection)
stop_detection_button.grid(row=10, column=3, padx=20, pady=5)

video_record_button = tk.Button(window, text="Записать видео", width=20, bg="green", fg="white", relief="solid", command=start_video_recording)
video_record_button.grid(row=7, column=4, padx=20, pady=5)

stop_video_record_button = tk.Button(window, text="Остановить запись", width=20, bg="red", fg="white", relief="solid", command=stop_video_recording)
stop_video_record_button.grid(row=8, column=4, padx=20, pady=5)

camera_label = tk.Label(window)
camera_label.grid(row=0, column=3, rowspan=8)

# Кнопки телеметрии
csv_telem_button = tk.Button(window, text="Запись телеметрии", width=20, bg="blue", fg="white", relief="solid", command=start_telemetry_recording)
csv_telem_button.grid(row=12, column=0, columnspan=2, padx=20, pady=5)

# Кнопка светодиодов
led_button = tk.Button(window, text="Светодиод", width=20, bg="yellow", fg="black", relief="solid", command=turn_on_led)
led_button.grid(row=9, column=4, columnspan=2, padx=20, pady=5)

# Кнопка построения графика
plot_button = tk.Button(window, text="Построить график", width=20, bg="silver", fg="black", relief="solid", command=plot_graph)
plot_button.grid(row=10, column=4, columnspan=2, padx=20, pady=5)

# Функции обновления и mainloop
window.after(100, update_image)
window.after(1000, update_altitude)
update_range_label()

window.mainloop()
