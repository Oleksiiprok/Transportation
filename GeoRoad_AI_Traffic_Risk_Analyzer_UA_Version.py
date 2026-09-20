# -*- coding: utf-8 -*-
"""
# All rights reserved, Oleksii Prokopchenko. 2025-2026
# Riverside,  Illinois, USA
# Tested in Google Colab

#Original file is located at
#    https://colab.research.google.com/drive/1y4sNSUlVpbFAXT6ZjiVNxGPpPJe5UZHr
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import folium
from IPython.display import HTML, display

# ==============================================================================
# ЕТАП 1: ГЕНЕРАЦІЯ ДАНИХ ТА ПІДГОТОВКА ДАТАСЕТУ
# ==============================================================================
data = {
    'Швидкість': np.random.uniform(30, 120, 1000),
    'Дорожні_умови': np.random.choice(['Сухо', 'Мокро', 'Сніг'], 1000, p=[0.7, 0.2, 0.1]),
    'Видимість': np.random.uniform(0.1, 10, 1000),
    'Ризик_аварії': np.zeros(1000)
}
df = pd.DataFrame(data)

df['Ризик_аварії'] = np.where(
    (df['Швидкість'] > 90) | (df['Дорожні_умови'].isin(['Мокро', 'Сніг'])),
    1, 0
)
df['Дорожні_умови_кодовані'] = df['Дорожні_умови'].astype('category').cat.codes

features = ['Швидкість', 'Видимість', 'Дорожні_умови_кодовані']
target = 'Ризик_аварії'

X = df[features]
y = df[target]

# ==============================================================================
# ЕТАП 2: НАВЧАННЯ МОДЕЛІ МАШИННОГО НАВЧАННЯ (RANDOM FOREST)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

# ==============================================================================
# ЕТАП 3: ГЕОПРОСТОРОВЕ ПРОГНОЗУВАННЯ ТА СТВОРЕННЯ МАПИ (КАЛІФОРНІЯ)
# ==============================================================================
df_map = pd.DataFrame({
    'lat': np.random.uniform(34.0, 34.5, 20),
    'lon': np.random.uniform(-118.5, -118.0, 20),
    'Швидкість': np.random.uniform(30, 120, 20),
    'Дорожні_умови': np.random.choice(['Сухо', 'Мокро'], 20, p=[0.8, 0.2]),
    'Видимість': np.random.uniform(0.1, 10, 20),
})

df_map['Дорожні_умови_кодовані'] = df_map['Дорожні_умови'].astype('category').cat.codes
df_map['Прогноз_ризику'] = model.predict(df_map[features])

# Центруємо мапу на Лос-Анджелесі, Каліфорнія (штучні координати для демонстрації геопросторового аналізу)
m = folium.Map(location=[34.05, -118.25], zoom_start=10, tiles='Esri.WorldStreetMap')

for _, row in df_map.iterrows():
    color = 'red' if row['Прогноз_ризику'] == 1 else 'green'
    tooltip = f"Швидкість: {row['Швидкість']:.1f} км/год<br>Умови: {row['Дорожні_умови']}<br>Ризик: {'Високий' if row['Прогноз_ризику'] == 1 else 'Низький'}"
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        tooltip=tooltip
    ).add_to(m)

# ==============================================================================
# ЕТАП 4: ФОРМУВАННЯ ІНТЕРФЕЙСУ ЗГІДНО З ВИМОГАМИ КОРИСТУВАЧА
# ==============================================================================

features_html = ""
for feat, imp in feature_importances.items():
    features_html += f"<li><b>{feat}</b>: {imp:.4f} (коефіцієнт важливості у прийнятті рішень)</li>"

dashboard_html = f"""
<div style='display: flex; gap: 12px; align-items: stretch; font-family: Arial, sans-serif; background: #ffffff; padding: 5px;'>

    <!-- ЛІВА КОЛОНКА: Зменшена ширина, збільшений шрифт, змінений порядок блоків -->
    <div style='flex: 0.65; background-color: #f8fafc; padding: 18px; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); box-sizing: border-box;'>

        <!-- БЛОК 1: Призначення програми -->
        <h3 style='color: #1e3a8a; margin-top: 0; border-bottom: 2px solid #cbd5e1; padding-bottom: 6px; font-size: 17px;'>📌 Призначення програми</h3>
        <p style='font-size: 13px; color: #334155; line-height: 1.5; margin: 8px 0 15px 0;'>
           Цей програмний комплекс розроблений для автоматизованого аналізу безпеки дорожнього руху та прогнозування виникнення аварійних ситуацій (ДТП) за допомогою методів штучного інтелекту.
           <br><br><b>Чому на мапі представлена саме Каліфорнія?</b> Для демонстрації геопросторових можливостей системи було обрано регіон Південної Каліфорнії (зокрема район Лос-Анджелеса). У межах цього регіону випадковим чином згенеровано координати контрольних точок (`lat`, `lon`), щоб показати, як модель аналізує реальні транспортні вузли незалежно від географії. У реальних проєктах сюди підключаються GPS-трекери автомобілів або відкриті бази даних міського трафіку (наприклад, OpenData).
        </p>

        <!-- БЛОК 2: Детальний аналітичний звіт -->
        <h3 style='color: #1e3a8a; border-bottom: 2px solid #cbd5e1; padding-bottom: 6px; font-size: 17px; margin-top: 20px;'>📊 Детальний аналітичний звіт</h3>

        <p style='margin: 10px 0 5px 0; font-size: 14px;'><b>1. Загальна точність моделі:</b> <span style='color: #059669; font-weight: bold;'>{accuracy * 100:.2f}%</span></p>
        <p style='font-size: 13px; color: #334155; line-height: 1.5; margin: 0 0 12px 0;'>
           <b>Як працює алгоритм RandomForest (Випадковий ліс):</b> Це потужний ансамблевий метод класифікації. Він будує сотню окремих «дерев рішень», кожне з яких оцінює дані під своїм кутом. У кінці проводиться колективне голосування: якщо більшість дерев вирішує, що комбінація швидкості та дощу призведе до аварії, система присвоює точці статус високого ризику.
           <br><br><b>Що таке тестові дані та звідки вони беруться:</b> Ми створили синтетичний датасет на 1000 записів із різними параметрами руху. За допомогою функції <code>train_test_split</code> ми розділили масив: 70% даних використали для «навчання» дерева рішень закономірностям, а решту 30% залишили як <i>тестові дані</i> (модель ніколи не бачила їх раніше). Перевіривши прогнози на цих 30%, ми порівняли їх з реальними мітками та отримали точність у <code>{accuracy * 100:.2f}%</code>.
        </p>

        <p style='margin: 10px 0 5px 0; font-size: 14px;'><b>2. Важливість факторів впливу (Feature Importances):</b></p>
        <ul style='margin: 4px 0 10px 0; padding-left: 18px; font-size: 13px; color: #334155; line-height: 1.5;'>
            {features_html}
        </ul>
        <p style='font-size: 12px; color: #475569; line-height: 1.5; margin: 0;'>
           <b>Детальне пояснення для користувача:</b> Ці показники демонструють ступінь впливу кожного параметра на кінцевий вердикт ШІ. Наприклад, найбільша вага параметра <b>Швидкість</b> означає, що перевищення ліміту є найвагомішим тригером небезпеки. Дорожні умови (мокро/сніг) також суттєво впливають на ймовірність аварії, тоді як видимість у даній моделі відіграє меншу роль.
        </p>
    </div>

    <!-- ПРАВА КОЛОНКА: Мапа, зсунута ліворуч та адаптована за висотою -->
    <div style='flex: 1.35; display: flex; flex-direction: column; border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05); box-sizing: border-box;'>
        <div style='background: #1e3a8a; color: white; padding: 10px 14px; font-size: 14px; font-weight: bold;'>
            🗺️ Інтерактивна мапа ризиків (регіон Каліфорнія)
        </div>
        <div style='flex-grow: 1; width: 100%; min-height: 650px;'>
            {m._repr_html_()}
        </div>
    </div>

</div>
"""

display(HTML(dashboard_html))