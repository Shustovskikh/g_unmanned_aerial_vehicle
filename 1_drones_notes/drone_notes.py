import datetime

def main_menu():
    """
    User interaction menu.
    To display a menu and request a user's choice.
    Depending on the choice, one of the functions will be called:
    add_note() or view_notes()
    """
    while True:
        print("\nМеню:")
        print("1. Добавить новую заметку о БПЛА")
        print("2. Просмотреть сохраненные заметки о БПЛА")
        print("3. Выход")
        choice = input("Выберите опцию (1, 2 или 3): ")

        if choice == '1':
            add_note()
        elif choice == '2':
            view_notes()
        elif choice == '3':
            print("Программа завершена.")
            break
        else:
            print("Неверный ввод. Попробуйте снова.")


def add_note():
    """
    Adding notes.
    Saves the note to a file notes.txt.
    If there is no file, it will be created automatically.
    """
    note = input("Введите текст заметки о БПЛА: ")
    if note.strip() == "":
        print("Заметка не может быть пустой.")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_note = f"{timestamp} - {note}\n"

    with open("notes.txt", "a", encoding="utf-8") as file:
        file.write(formatted_note)

    print("Заметка успешно сохранена!")

def view_notes():
    """
    Viewing notes.
    1. There is no file, the program displays a message that there are no notes yet.
    2. The file exists, all lines (notes) are read and output.
    """
    try:
        with open("notes.txt", "r", encoding="utf-8") as file:
            notes = file.readlines()
            if notes:
                print("\nСохраненные заметки о БПЛА:")
                for note in notes:
                    print(note.strip())
            else:
                print("Заметок пока нет.")
    except FileNotFoundError:
        print("Файл с заметками не найден. Добавьте первую заметку о БПЛА.")

if __name__ == "__main__":
    main_menu()