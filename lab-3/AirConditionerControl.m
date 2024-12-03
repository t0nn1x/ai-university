% Створення нової нечіткої системи типу Мандані
fis = mamfis('Name', 'AirConditionerControl');

% Додавання вхідної змінної 'AirTemperature'
fis = addInput(fis, [10 35], 'Name', 'AirTemperature');

% Додавання функцій належності для 'AirTemperature'
fis = addMF(fis, 'AirTemperature', 'trapmf', [10 10 12 16], 'Name', 'VeryCold');
fis = addMF(fis, 'AirTemperature', 'trimf', [14 18 22], 'Name', 'Cold');
fis = addMF(fis, 'AirTemperature', 'trimf', [20 23 26], 'Name', 'Normal');
fis = addMF(fis, 'AirTemperature', 'trimf', [24 28 32], 'Name', 'Warm');
fis = addMF(fis, 'AirTemperature', 'trapmf', [30 33 35 35], 'Name', 'VeryWarm');

% Додавання вхідної змінної 'TemperatureChangeRate'
fis = addInput(fis, [-5 5], 'Name', 'TemperatureChangeRate');

% Додавання функцій належності для 'TemperatureChangeRate'
fis = addMF(fis, 'TemperatureChangeRate', 'trapmf', [-5 -5 -1 0], 'Name', 'Negative');
fis = addMF(fis, 'TemperatureChangeRate', 'trimf', [-0.5 0 0.5], 'Name', 'Zero');
fis = addMF(fis, 'TemperatureChangeRate', 'trapmf', [0 1 5 5], 'Name', 'Positive');

% Додавання вихідної змінної 'RegulatorAngle'
fis = addOutput(fis, [-100 100], 'Name', 'RegulatorAngle');

% Додавання функцій належності для 'RegulatorAngle'
fis = addMF(fis, 'RegulatorAngle', 'trapmf', [-100 -100 -75 -50], 'Name', 'LargeLeft');
fis = addMF(fis, 'RegulatorAngle', 'trimf', [-60 -40 -20], 'Name', 'SmallLeft');
fis = addMF(fis, 'RegulatorAngle', 'trimf', [-5 0 5], 'Name', 'Zero');
fis = addMF(fis, 'RegulatorAngle', 'trimf', [20 40 60], 'Name', 'SmallRight');
fis = addMF(fis, 'RegulatorAngle', 'trapmf', [50 75 100 100], 'Name', 'LargeRight');

% Правила керування
rules = [
    % [AirTemperature   TemperatureChangeRate   RegulatorAngle   Weight   Operator]
    % Правило 1
    5 3 1 1 1; % Якщо Дуже Тепла і Додатна, то Великий Вліво
    % Правило 2
    5 1 2 1 1; % Якщо Дуже Тепла і Відʼємна, то Невеликий Вліво
    % Правило 3
    4 3 1 1 1; % Якщо Тепла і Додатна, то Великий Вліво
    % Правило 4
    4 1 3 1 1; % Якщо Тепла і Відʼємна, то Виключити
    % Правило 5
    1 1 5 1 1; % Якщо Дуже Холодна і Відʼємна, то Великий Вправо
    % Правило 6
    1 3 4 1 1; % Якщо Дуже Холодна і Додатна, то Невеликий Вправо
    % Правило 7
    2 1 5 1 1; % Якщо Холодна і Відʼємна, то Великий Вправо
    % Правило 8
    2 3 3 1 1; % Якщо Холодна і Додатна, то Виключити
    % Правило 9
    5 2 1 1 1; % Якщо Дуже Тепла і Нульова, то Великий Вліво
    % Правило 10
    4 2 2 1 1; % Якщо Тепла і Нульова, то Невеликий Вліво
    % Правило 11
    1 2 5 1 1; % Якщо Дуже Холодна і Нульова, то Великий Вправо
    % Правило 12
    2 2 4 1 1; % Якщо Холодна і Нульова, то Невеликий Вправо
    % Правило 13
    3 3 2 1 1; % Якщо Нормальна і Додатна, то Невеликий Вліво
    % Правило 14
    3 1 4 1 1; % Якщо Нормальна і Відʼємна, то Невеликий Вправо
    % Правило 15
    3 2 3 1 1; % Якщо Нормальна і Нульова, то Виключити
];

% Додавання правил до FIS
fis = addRule(fis, rules);

% Відкрити переглядач правил
ruleview(fis);

% Приклад вхідних значень
% Температура повітря = 28°C, Швидкість зміни температури = 1°C/од.часу
input = [28 1];

% Обчислення виходу
output = evalfis(fis, input);

% Відображення результату
RegulatorAngle = output
