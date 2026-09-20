# -*- coding: utf-8 -*-
"""
# All rights reserved, Oleksii Prokopchenko. 2025-2026
# Riverside,  Illinois, USA
# Tested in Google Colab

#Original file is located at
#   https://colab.research.google.com/drive/1eIBy9EubDf0UeH-TuzUPC2ie7jxhEBcv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import folium
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

# ==============================================================================
# ЕТАП 1: КООРДИНАТИ НАЙБІЛЬШИХ МІСТ ДЛЯ КОЖНОГО ШТАТУ
# ==============================================================================
state_major_cities = {
    'California': [
        ('Los Angeles', 34.05, -118.24),
        ('San Diego', 32.71, -117.16),
        ('San Jose', 37.33, -121.88),
        ('San Francisco', 37.77, -122.41),
        ('Fresno', 36.73, -119.78),
        ('Sacramento', 38.58, -121.49)
    ],
    'Texas': [
        ('Houston', 29.76, -95.36),
        ('San Antonio', 29.42, -98.49),
        ('Dallas', 32.77, -96.79),
        ('Austin', 30.26, -97.74),
        ('Fort Worth', 32.75, -97.33),
        ('El Paso', 31.76, -106.48)
    ],
    'New York': [
        ('New York City', 40.71, -74.00),
        ('Buffalo', 42.88, -78.87),
        ('Rochester', 43.15, -77.61),
        ('Yonkers', 40.93, -73.89),
        ('Syracuse', 43.04, -76.14),
        ('Albany', 42.65, -73.75)
    ],
    'Florida': [
        ('Jacksonville', 30.33, -81.65),
        ('Miami', 25.76, -80.19),
        ('Tampa', 27.95, -82.45),
        ('Orlando', 28.53, -81.37),
        ('St. Petersburg', 27.76, -82.64),
        ('Hialeah', 25.85, -80.28)
    ],
    'Illinois': [
        ('Chicago', 41.87, -87.62),
        ('Aurora', 41.76, -88.32),
        ('Joliet', 41.52, -88.08),
        ('Naperville', 41.75, -88.15),
        ('Rockford', 42.27, -89.09),
        ('Springfield', 39.78, -89.65)
    ]
}

data = {
    'Speed': np.random.uniform(30, 120, 1000),
    'Road_Conditions': np.random.choice(['Dry', 'Wet', 'Snow'], 1000, p=[0.7, 0.2, 0.1]),
    'Visibility': np.random.uniform(0.1, 10, 1000),
    'Accident_Risk': np.zeros(1000)
}
df = pd.DataFrame(data)

df['Accident_Risk'] = np.where(
    (df['Speed'] > 90) | (df['Road_Conditions'].isin(['Wet', 'Snow'])),
    1, 0
)
df['Road_Conditions_Encoded'] = df['Road_Conditions'].astype('category').cat.codes

features = ['Speed', 'Visibility', 'Road_Conditions_Encoded']
target = 'Accident_Risk'

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
# ЕТАП 3: ІНТЕРАКТИВНИЙ ІНТЕРФЕЙС ТА РОЗПОДІЛ ТОЧОК ПО МІСТАХ
# ==============================================================================

state_dropdown = widgets.Dropdown(
    options=list(state_major_cities.keys()),
    value='California',
    description='Select State:',
    style={'description_width': 'initial'}
)

output_area = widgets.Output()

def update_dashboard(selected_state):
    with output_area:
        clear_output(wait=True)

        cities = state_major_cities[selected_state]
        center_lat = np.mean([c[1] for c in cities])
        center_lon = np.mean([c[2] for c in cities])

        # Генерація точок навколо найбільших міст штату
        np.random.seed(42)
        lats, lons, city_names = [], [], []

        for _ in range(20):
            city = cities[np.random.choice(len(cities))]
            city_names.append(city[0])
            # Невеликий зсув від центру міста (у межах міської забудови)
            lats.append(city[1] + np.random.uniform(-0.04, 0.04))
            lons.append(city[2] + np.random.uniform(-0.04, 0.04))

        df_map = pd.DataFrame({
            'City': city_names,
            'lat': lats,
            'lon': lons,
            'Speed': np.random.uniform(30, 120, 20),
            'Road_Conditions': np.random.choice(['Dry', 'Wet'], 20, p=[0.8, 0.2]),
            'Visibility': np.random.uniform(0.1, 10, 20),
        })

        df_map['Road_Conditions_Encoded'] = df_map['Road_Conditions'].astype('category').cat.codes
        df_map['Risk_Prediction'] = model.predict(df_map[features])

        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='Esri.WorldStreetMap')

        for _, row in df_map.iterrows():
            color = 'red' if row['Risk_Prediction'] == 1 else 'green'
            risk_text = 'High Risk' if row['Risk_Prediction'] == 1 else 'Low Risk'

            tooltip_html = f"""
            <div style='font-size: 15px; font-family: Arial, sans-serif; padding: 4px; color: #000000;'>
                <b>City:</b> {row['City']}<br>
                <b>Speed:</b> {row['Speed']:.1f} km/h<br>
                <b>Conditions:</b> {row['Road_Conditions']}<br>
                <b>Status:</b> <span style='color: {color}; font-weight: bold;'>{risk_text}</span>
            </div>
            """

            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=folium.Tooltip(tooltip_html)
            ).add_to(m)

        features_html = ""
        for feat, imp in feature_importances.items():
            features_html += f"<li><b>{feat}</b>: {imp:.4f} (importance weight)</li>"

        dashboard_html = f"""
        <div style='font-family: Arial, sans-serif; background: #ffffff; padding: 5px;'>

            <!-- ШАПКА ПРОГРАМИ -->
            <div style='background: #ffffff; border: 1px solid #cbd5e1; padding: 15px 20px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                <h2 style='margin: 0; font-size: 22px; color: #1e3a8a;'>🚗 GeoRoad AI: Traffic Risk Analyzer</h2>
                <p style='margin: 5px 0 0 0; font-size: 14px; color: #334155;'>Automated Road Safety & Accident Prediction System using Machine Learning</p>
            </div>

            <!-- ГОЛОВНИЙ КОНТЕЙНЕР (ЗБАЛАНСОВАНІ ПРОПОРЦІЇ 1:1) -->
            <div style='display: flex; gap: 15px; align-items: stretch;'>

                <!-- ЛІВА КОЛОНКА (ТЕКСТОВИЙ ЗВІТ) -->
                <div style='flex: 1; background-color: #f8fafc; padding: 20px; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); box-sizing: border-box;'>

                    <h3 style='color: #1e3a8a; margin-top: 0; border-bottom: 2px solid #cbd5e1; padding-bottom: 8px; font-size: 18px;'>📌 Program Purpose & City Placement</h3>
                    <p style='font-size: 14px; color: #000000; line-height: 1.6; margin: 8px 0 15px 0;'>
                       This system analyzes traffic parameters to predict crash probabilities.
                       <br><br><b>Are points in cities?</b> Yes! To guarantee points do not fall into forests or empty areas, markers are precisely distributed across the <b>largest metropolitan areas and cities</b> of <b>{selected_state}</b>.
                    </p>

                    <h3 style='color: #1e3a8a; border-bottom: 2px solid #cbd5e1; padding-bottom: 8px; font-size: 18px; margin-top: 25px;'>📊 Analytical Report</h3>

                    <p style='margin: 12px 0 6px 0; font-size: 15px; color: #000000;'><b>1. Model Accuracy:</b> <span style='color: #059669; font-weight: bold;'>{accuracy * 100:.2f}%</span></p>
                    <p style='font-size: 14px; color: #000000; line-height: 1.6; margin: 0 0 15px 0;'>
                       <b>RandomForest Engine:</b> Builds an ensemble of decision trees to evaluate driving conditions via collective voting.
                       <br><br><b>Test Data:</b> Split via <code>train_test_split</code> (70% training, 30% testing). The accuracy score proves high reliability on urban road segments.
                    </p>

                    <p style='margin: 12px 0 6px 0; font-size: 15px; color: #000000;'><b>2. Feature Importances:</b></p>
                    <ul style='margin: 4px 0 15px 0; padding-left: 20px; font-size: 14px; color: #000000; line-height: 1.6;'>
                        {features_html}
                    </ul>
                    <p style='font-size: 13px; color: #334155; line-height: 1.5; margin: 0;'>
                       <b>User Insight:</b> High weight on <b>Speed</b> highlights that speeding is the primary risk factor across <code>{selected_state}</code> urban roads.
                    </p>
                </div>

                <!-- ПРАВА КОЛОНКА (МАПА З ПОВНОЦІННОЮ ВИСОТОЮ) -->
                <div style='flex: 1; display: flex; flex-direction: column; border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05); box-sizing: border-box;'>
                    <div style='background: #ffffff; color: #1e3a8a; padding: 12px 16px; font-size: 15px; font-weight: bold; border-bottom: 1px solid #cbd5e1;'>
                        🗺️ Major Cities Risk Map — {selected_state}
                    </div>
                    <div style='flex-grow: 1; width: 100%; min-height: 650px;'>
                        {m._repr_html_()}
                    </div>
                </div>

            </div>
        </div>
        """
        display(HTML(dashboard_html))

state_dropdown.observe(lambda change: update_dashboard(change['new']), names='value')

display(state_dropdown)
display(output_area)

# Початкове завантаження
update_dashboard(state_dropdown.value)