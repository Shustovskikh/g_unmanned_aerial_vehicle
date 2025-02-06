import tkinter as tk
from tkinter import messagebox

window = tk.Tk()
window.title("Маршрут БПЛА")
window.resizable(False, False)
window.geometry("400x700")

tk.Label(window, text="Начальная точка:").pack()
tk.Label(window, text="Широта:").pack()
entry_start_lat = tk.Entry(window)
entry_start_lat.pack()

tk.Label(window, text="Долгота:").pack()
entry_start_lon = tk.Entry(window)
entry_start_lon.pack()

tk.Label(window, text="Конечная точка:").pack()
tk.Label(window, text="Широта:").pack()
entry_end_lat = tk.Entry(window)
entry_end_lat.pack()

tk.Label(window, text="Долгота:").pack()
entry_end_lon = tk.Entry(window)
entry_end_lon.pack()

tk.Label(window, text="Промежуточная точка:").pack()
tk.Label(window, text="Широта:").pack()
entry_inter_lat = tk.Entry(window)
entry_inter_lat.pack()

tk.Label(window, text="Долгота:").pack()
entry_inter_lon = tk.Entry(window)
entry_inter_lon.pack()

tk.Label(window, text="Описание точки:").pack()
entry_description = tk.Entry(window)
entry_description.pack()

intermediate_points = []
listbox_points = tk.Listbox(window)
listbox_points.pack()

def add_intermediate_point():
    """
    description in the list widget
    """
    try:
        lat = float(entry_inter_lat.get())
        lon = float(entry_inter_lon.get())
        description = entry_description.get() or "Без описания"
        point = {'lat': lat, 'lon': lon, 'description': description}
        intermediate_points.append(point)
        listbox_points.insert(tk.END, f"{lat}, {lon} - {description}")
        entry_inter_lat.delete(0, tk.END)
        entry_inter_lon.delete(0, tk.END)
        entry_description.delete(0, tk.END)
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректные координаты!")

def remove_selected_point():
    """
    deleting a selected point
    """
    selected_index = listbox_points.curselection()
    if selected_index:
        index = selected_index[0]
        listbox_points.delete(index)
        intermediate_points.pop(index)
    else:
        messagebox.showwarning("Ошибка", "Выберите точку для удаления!")

def edit_selected_point():
    """
    editing the selected point
    """
    selected_index = listbox_points.curselection()
    if selected_index:
        index = selected_index[0]
        point = intermediate_points[index]
        entry_inter_lat.delete(0, tk.END)
        entry_inter_lon.delete(0, tk.END)
        entry_description.delete(0, tk.END)
        entry_inter_lat.insert(0, str(point['lat']))
        entry_inter_lon.insert(0, str(point['lon']))
        entry_description.insert(0, point['description'])
        button_add_point.config(text="Сохранить изменения", command=lambda: save_point(index))
    else:
        messagebox.showwarning("Ошибка", "Выберите точку для редактирования!")

def save_point(index):
    """
    saving changes
    """
    try:
        lat = float(entry_inter_lat.get())
        lon = float(entry_inter_lon.get())
        description = entry_description.get() or "Без описания"
        intermediate_points[index] = {'lat': lat, 'lon': lon, 'description': description}
        listbox_points.delete(index)
        listbox_points.insert(index, f"{lat}, {lon} - {description}")
        button_add_point.config(text="Добавить точку", command=add_intermediate_point)
        entry_inter_lat.delete(0, tk.END)
        entry_inter_lon.delete(0, tk.END)
        entry_description.delete(0, tk.END)
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректные координаты!")

def calculate_route():
    try:
        start_lat = float(entry_start_lat.get())
        start_lon = float(entry_start_lon.get())
        end_lat = float(entry_end_lat.get())
        end_lon = float(entry_end_lon.get())
        route = [(start_lat, start_lon)] + [(pt['lat'], pt['lon']) for pt in intermediate_points] + [(end_lat, end_lon)]
        route_str = " -> ".join([f"({lat}, {lon})" for lat, lon in route])
        result_label.config(text=f"Маршрут: {route_str}")
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректные координаты для начальной и конечной точек!")

button_add_point = tk.Button(window, text="Добавить точку", command=add_intermediate_point)
button_add_point.pack()

button_remove_point = tk.Button(window, text="Удалить точку", command=remove_selected_point)
button_remove_point.pack()

button_edit_point = tk.Button(window, text="Редактировать точку", command=edit_selected_point)
button_edit_point.pack()

button_calculate = tk.Button(window, text="Рассчитать маршрут", command=calculate_route)
button_calculate.pack()

button_exit = tk.Button(window, text="Выйти", command=window.quit)
button_exit.pack()

result_label = tk.Label(window, text="", fg="green")
result_label.pack()

window.mainloop()
