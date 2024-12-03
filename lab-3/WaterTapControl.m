% Створення нової нечіткої системи типу Мандані
fis = mamfis('Name', 'WaterTapControl');

% Додавання вхідної змінної 'WaterTemperature'
fis = addInput(fis, [0 50], 'Name', 'WaterTemperature');

% Додавання функцій належності для 'WaterTemperature'
fis = addMF(fis, 'WaterTemperature', 'trapmf', [0 0 5 15], 'Name', 'Cold');
fis = addMF(fis, 'WaterTemperature', 'trimf', [10 17.5 25], 'Name', 'Cool');
fis = addMF(fis, 'WaterTemperature', 'trimf', [20 27.5 35], 'Name', 'Warm');
fis = addMF(fis, 'WaterTemperature', 'trimf', [30 37.5 45], 'Name', 'NotVeryHot');
fis = addMF(fis, 'WaterTemperature', 'trapmf', [40 45 50 50], 'Name', 'Hot');

% Додавання вхідної змінної 'WaterPressure'
fis = addInput(fis, [0 100], 'Name', 'WaterPressure');

% Додавання функцій належності для 'WaterPressure'
fis = addMF(fis, 'WaterPressure', 'trapmf', [0 0 20 40], 'Name', 'Weak');
fis = addMF(fis, 'WaterPressure', 'trimf', [30 50 70], 'Name', 'NotVeryStrong');
fis = addMF(fis, 'WaterPressure', 'trapmf', [60 80 100 100], 'Name', 'Strong');

% Додавання вихідної змінної 'AngleHotTap'
fis = addOutput(fis, [-90 90], 'Name', 'AngleHotTap');

% Додавання функцій належності для 'AngleHotTap'
fis = addMF(fis, 'AngleHotTap', 'trapmf', [-90 -90 -75 -45], 'Name', 'LargeLeft');
fis = addMF(fis, 'AngleHotTap', 'trimf', [-60 -45 -30], 'Name', 'MediumLeft');
fis = addMF(fis, 'AngleHotTap', 'trimf', [-35 -20 -5], 'Name', 'SmallLeft');
fis = addMF(fis, 'AngleHotTap', 'trimf', [-5 0 5], 'Name', 'Zero');
fis = addMF(fis, 'AngleHotTap', 'trimf', [5 20 35], 'Name', 'SmallRight');
fis = addMF(fis, 'AngleHotTap', 'trimf', [30 45 60], 'Name', 'MediumRight');
fis = addMF(fis, 'AngleHotTap', 'trapmf', [45 75 90 90], 'Name', 'LargeRight');

% Додавання вихідної змінної 'AngleColdTap'
fis = addOutput(fis, [-90 90], 'Name', 'AngleColdTap');

% Додавання функцій належності для 'AngleColdTap'
fis = addMF(fis, 'AngleColdTap', 'trapmf', [-90 -90 -75 -45], 'Name', 'LargeLeft');
fis = addMF(fis, 'AngleColdTap', 'trimf', [-60 -45 -30], 'Name', 'MediumLeft');
fis = addMF(fis, 'AngleColdTap', 'trimf', [-35 -20 -5], 'Name', 'SmallLeft');
fis = addMF(fis, 'AngleColdTap', 'trimf', [-5 0 5], 'Name', 'Zero');
fis = addMF(fis, 'AngleColdTap', 'trimf', [5 20 35], 'Name', 'SmallRight');
fis = addMF(fis, 'AngleColdTap', 'trimf', [30 45 60], 'Name', 'MediumRight');
fis = addMF(fis, 'AngleColdTap', 'trapmf', [45 75 90 90], 'Name', 'LargeRight');

% Визначення списку правил
ruleList = [
    5 3 2 6 1 1; % Правило 1
    5 2 4 6 1 1; % Правило 2
    4 3 3 4 1 1; % Правило 3
    4 1 5 5 1 1; % Правило 4
    3 2 4 4 1 1; % Правило 5
    2 3 6 2 1 1; % Правило 6
    2 2 6 3 1 1; % Правило 7
    1 1 7 4 1 1; % Правило 8
    1 3 2 6 1 1; % Правило 9
    3 3 3 3 1 1; % Правило 10
    3 1 5 5 1 1; % Правило 11
];

% Додавання правил до FIS
fis = addRule(fis, ruleList);

% Відкрити переглядач правил
ruleview(fis);

% Збереження нечіткої системи в файл
writeFIS(fis, 'WaterTapControl.fis');
