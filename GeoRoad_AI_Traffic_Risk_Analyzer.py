# -*- coding: utf-8 -*-
"""
# All rights reserved, Oleksii Prokopchenko. 2025-2026
# Riverside,  Illinois, USA
# Tested in Google Colab
#Original file is located at
#    https://colab.research.google.com/drive/1NpwLNRJzYonipAeOO6rtzvgrYmUH54vB
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import folium
from IPython.display import HTML, display

# ==============================================================================
# STEP 1: DATA GENERATION AND DATASET PREPARATION
# ==============================================================================
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
# STEP 2: MACHINE LEARNING MODEL TRAINING (RANDOM FOREST)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

# ==============================================================================
# STEP 3: GEOSPATIAL FORECASTING AND MAP CREATION (CALIFORNIA)
# ==============================================================================
df_map = pd.DataFrame({
    'lat': np.random.uniform(34.0, 34.5, 20),
    'lon': np.random.uniform(-118.5, -118.0, 20),
    'Speed': np.random.uniform(30, 120, 20),
    'Road_Conditions': np.random.choice(['Dry', 'Wet'], 20, p=[0.8, 0.2]),
    'Visibility': np.random.uniform(0.1, 10, 20),
})

df_map['Road_Conditions_Encoded'] = df_map['Road_Conditions'].astype('category').cat.codes
df_map['Risk_Prediction'] = model.predict(df_map[features])

# Center the map on Los Angeles, California (synthetic coordinates for geospatial demo)
m = folium.Map(location=[34.05, -118.25], zoom_start=10, tiles='Esri.WorldStreetMap')

for _, row in df_map.iterrows():
    color = 'red' if row['Risk_Prediction'] == 1 else 'green'
    tooltip = f"Speed: {row['Speed']:.1f} km/h<br>Conditions: {row['Road_Conditions']}<br>Risk: {'High' if row['Risk_Prediction'] == 1 else 'Low'}"
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        tooltip=tooltip
    ).add_to(m)

# ==============================================================================
# STEP 4: INTERFACE GENERATION WITH ENGLISH LOCALIZATION
# ==============================================================================

features_html = ""
for feat, imp in feature_importances.items():
    features_html += f"<li><b>{feat}</b>: {imp:.4f} (importance weight in model decision-making)</li>"

dashboard_html = f"""
<div style='display: flex; gap: 12px; align-items: stretch; font-family: Arial, sans-serif; background: #ffffff; padding: 5px;'>

    <!-- LEFT COLUMN: Compact width, larger font, purpose first -->
    <div style='flex: 0.65; background-color: #f8fafc; padding: 18px; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); box-sizing: border-box;'>

        <!-- BLOCK 1: Program Purpose -->
        <h3 style='color: #1e3a8a; margin-top: 0; border-bottom: 2px solid #cbd5e1; padding-bottom: 6px; font-size: 17px;'>📌 Program Purpose</h3>
        <p style='font-size: 13px; color: #334155; line-height: 1.5; margin: 8px 0 15px 0;'>
           This software package is designed for automated traffic safety analysis and predicting accident (crash) risks using artificial intelligence methods.
           <br><br><b>Why is California shown on the map?</b> To demonstrate geospatial capabilities, the Southern California region (specifically the Los Angeles area) was selected. Control point coordinates (<code>lat</code>, <code>lon</code>) are randomly generated within this region to show how the model analyzes traffic nodes regardless of geography. In real-world projects, this connects to vehicle GPS trackers or open urban traffic databases (e.g., OpenData).
        </p>

        <!-- BLOCK 2: Detailed Analytical Report -->
        <h3 style='color: #1e3a8a; border-bottom: 2px solid #cbd5e1; padding-bottom: 6px; font-size: 17px; margin-top: 20px;'>📊 Detailed Analytical Report</h3>

        <p style='margin: 10px 0 5px 0; font-size: 14px;'><b>1. Overall Model Accuracy:</b> <span style='color: #059669; font-weight: bold;'>{accuracy * 100:.2f}%</span></p>
        <p style='font-size: 13px; color: #334155; line-height: 1.5; margin: 0 0 12px 0;'>
           <b>How the RandomForest Algorithm Works:</b> This is a powerful ensemble classification method. It builds hundreds of individual decision trees, each evaluating data from its own perspective. Finally, collective voting takes place: if most trees decide that a combination of speed and rain leads to a crash, the system flags the point as high risk.
           <br><br><b>What are test data and where do they come from?</b> We created a synthetic dataset of 1,000 records with various driving parameters. Using the <code>train_test_split</code> function, we split the array: 70% was used to "train" the decision trees on patterns, while the remaining 30% was set aside as <i>test data</i> (unseen by the model). Testing predictions on this 30% and comparing them with actual labels yielded an accuracy of <code>{accuracy * 100:.2f}%</code>.
        </p>

        <p style='margin: 10px 0 5px 0; font-size: 14px;'><b>2. Feature Importances:</b></p>
        <ul style='margin: 4px 0 10px 0; padding-left: 18px; font-size: 13px; color: #334155; line-height: 1.5;'>
            {features_html}
        </ul>
        <p style='font-size: 12px; color: #475569; line-height: 1.5; margin: 0;'>
           <b>Detailed User Explanation:</b> These metrics demonstrate the influence level of each parameter on the final AI verdict. For instance, the highest weight on <b>Speed</b> indicates that exceeding speed limits is the most significant trigger for danger. Road conditions (wet/snow) also substantially impact crash probability, whereas visibility plays a minor role in this model.
        </p>
    </div>

    <!-- RIGHT COLUMN: Map shifted left and height-adapted -->
    <div style='flex: 1.35; display: flex; flex-direction: column; border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05); box-sizing: border-box;'>
        <div style='background: #1e3a8a; color: white; padding: 10px 14px; font-size: 14px; font-weight: bold;'>
            🗺️ Interactive Risk Map (California Region)
        </div>
        <div style='flex-grow: 1; width: 100%; min-height: 650px;'>
            {m._repr_html_()}
        </div>
    </div>

</div>
"""

display(HTML(dashboard_html))