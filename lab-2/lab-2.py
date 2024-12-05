class Student:
    def __init__(self, first_name, last_name, student_id, major):
        self.first_name = first_name
        self.last_name = last_name
        self.student_id = student_id
        self.major = major

    # Метод, що дозволяє вивести на екран вітання від об’єкту класу
    def greet(self):
        print(f"Привіт, мене звати {self.first_name} {self.last_name}!")
    
    # Метод, що дозволяє вивести на екран інформацію про студента
    def display_info(self):
        print(f"Студент: {self.first_name} {self.last_name}")
        print(f"ID: {self.student_id}")
        print(f"Спеціальність: {self.major}")

# Створення об'єкта класу Student
student1 = Student("Олена", "Коваленко", "S12345", "Прикладна лінгвістика")

# Виклик методу greet
student1.greet()

# Виклик методу display_info
student1.display_info()

# Створення другого об'єкта класу Student
student2 = Student("Іван", "Петренко", "S67890", "Комп'ютерні науки")

# Створення третього об'єкта класу Student
student3 = Student("Марія", "Шевченко", "S54321", "Філологія")

# Виклик методів для student2
student2.greet()
student2.display_info()

# Виклик методів для student3
student3.greet()
student3.display_info()
