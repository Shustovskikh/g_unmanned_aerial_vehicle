import rospy
from clover import srv
from std_srvs.srv import Trigger
import math
import json
from datetime import datetime

rospy.init_node('drone_control')

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
land = rospy.ServiceProxy('land', Trigger)

flight_report = {
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "events": []
}

def log_event(action, details=None):
    """
    adding an entry to the report
    """
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details if details else {}
    }
    flight_report["events"].append(event)

def takeoff(navigate):
    """
    the drone takeoff
    """
    print("Взлет на высоту 3 метра...")
    try:
        navigate(x=0, y=0, z=3, frame_id='body', auto_arm=True)
        if wait_arrival():
            print("Дрон взлетел")
            log_event("takeoff", {"altitude": 3})
            return True
        else:
            print("Ошибка: дрон не достиг высоты 3 метра")
            log_event("takeoff_failed", {"altitude": 3})
            return False
    except rospy.ServiceException as e:
        print(f"Ошибка взлета: {e}")
        log_event("takeoff_error", {"error": str(e)})
        return False

def land_drone():
    """
    landing the drone
    """
    try:
        print("Приземляемся...")
        land()
        print("Дрон приземлился")
        log_event("land")
        return True
    except rospy.ServiceException as e:
        print(f"Ошибка при посадке: {e}")
        log_event("land_error", {"error": str(e)})
        return False

def fly_to_local_coordinates(navigate):
    """
    the drone flight by local coordinates
    """
    try:
        x = float(input("Введите координаты X (метры): "))
        y = float(input("Введите координаты Y (метры): "))
    except ValueError:
        print("Ошибка: пожалуйста, введите числовые значения")
        return
    print(f"Полет в точку X={x}, Y={y}")
    try:
        navigate(x=x, y=y, z=0, frame_id='body', speed=1)
        if wait_arrival():
            print("Дрон достиг точки по локальным координатам")
            log_event("fly_to_local_coordinates", {"x": x, "y": y})
        else:
            print("Ошибка: дрон не достиг заданной точки")
            log_event("fly_to_local_coordinates_failed", {"x": x, "y": y})
    except rospy.ServiceException as e:
        print(f"Ошибка полета по локальным координатам: {e}")
        log_event("fly_to_local_coordinates_error", {"x": x, "y": y, "error": str(e)})

def fly_to_global_coordinates(navigate_global):
    """
    the drone flight by global coordinates
    """
    try:
        lat = float(input("Введите широту: "))
        lon = float(input("Введите долготу: "))
    except ValueError:
        print("Ошибка: пожалуйста, введите числовые значения")
        return
    print(f"Полет в точку: Latitude={lat}, Longitude={lon}")
    try:
        navigate_global(lat=lat, lon=lon, z=3, yaw=math.inf, speed=1)
        if wait_arrival():
            print("Дрон достиг точки по глобальным координатам")
            log_event("fly_to_global_coordinates", {"latitude": lat, "longitude": lon})
        else:
            print("Ошибка: дрон не достиг заданной глобальной точки")
            log_event("fly_to_global_coordinates_failed", {"latitude": lat, "longitude": lon})
    except rospy.ServiceException as e:
        print(f"Ошибка полета по глобальным координатам: {e}")
        log_event("fly_to_global_coordinates_error", {"latitude": lat, "longitude": lon, "error": str(e)})

def wait_arrival(tolerance=0.2, timeout=30):
    """
    waiting for arrival at the target point with a timeout
    """
    start_time = rospy.get_time()
    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            return True
        if rospy.get_time() - start_time > timeout:
            print("Тайм-аут: дрон не достиг целевой точки")
            return False
        rospy.sleep(0.2)

def check_services():
    """
    checking the availability of services
    """
    try:
        get_telemetry.wait_for_service(timeout=5)
        navigate.wait_for_service(timeout=5)
        navigate_global.wait_for_service(timeout=5)
        land.wait_for_service(timeout=5)
        print("Все сервисы успешно подключены.")
    except rospy.ROSException as e:
        print(f"Ошибка подключения к сервисам: {e}")
        return False
    return True

def main():
    """
    the user's drone control program
    """
    is_flying = False
    while True:
        print("\nВыберите действие:")
        print("1. Взлет (высота 3 метра)")
        print("2. Приземление")
        print("3. Полет по локальным координатам")
        print("4. Полет по глобальным координатам")
        print("0. Выход")

        choice = input("Введите номер действия: ")

        if choice == '1':
            if not is_flying:
                is_flying = takeoff(navigate)
            else:
                print("Дрон уже в воздухе")

        elif choice == '2':
            if is_flying:
                is_flying = not land_drone()
            else:
                print("Дрон уже на земле")

        elif choice == '3':
            if is_flying:
                fly_to_local_coordinates(navigate)
            else:
                print("Сначала нужно взлететь")

        elif choice == '4':
            if is_flying:
                fly_to_global_coordinates(navigate_global)
            else:
                print("Сначала нужно взлететь")

        elif choice == '0':
            print("Выход из программы")
            if is_flying:
                land_drone()
            break
        else:
            print("Неверный код команды")

def save_report():
    """
    saving a report to a file
    """
    flight_report["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("flight_report.json", "w") as file:
        json.dump(flight_report, file, indent=4, ensure_ascii=False)
    print("Отчет сохранен в 'flight_report.json'")

if __name__ == "__main__":
    if check_services():
        main()
        save_report()
    else:
        print("Не удалось подключиться к необходимым сервисам!")
