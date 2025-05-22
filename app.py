import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 📦 Загрузка модели и вспомогательных файлов
model = joblib.load('model/rf_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')
symptom_list = joblib.load('model/symptom_list.pkl')

# 📄 Загрузка CSV
precautions_df = pd.read_csv('data/symptom_precaution.csv')
desc_df = pd.read_csv('data/symptom_Description.csv')
severity_df = pd.read_csv('data/Symptom-severity.csv')
severity_df['Symptom'] = severity_df['Symptom'].str.strip().str.lower()

# ✅ Функция очистки симптомов
def clear_symptoms():
    st.session_state["symptoms"] = []

# 🖼 Интерфейс
st.set_page_config(page_title="Предсказание болезни", layout="centered")
st.title("🩺 Предсказание болезни по симптомам")
st.markdown("Выберите симптомы, которые вы наблюдаете у себя:")

# 🧾 Мультивыбор симптомов
selected_symptoms = st.multiselect(
    "Симптомы:",
    symptom_list,
    key="symptoms"
)

# 🔁 Кнопка очистки выбора
st.button("🔁 Очистить", on_click=clear_symptoms)

# 🧠 Подготовка вектора признаков
input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

# 🔍 Кнопка предсказания
if st.button("🔍 Предсказать"):
    if not selected_symptoms:
        st.warning("Пожалуйста, выберите хотя бы один симптом.")
    else:
        # 🎯 Предсказание
        probs = model.predict_proba([input_vector])[0]
        prediction = model.predict([input_vector])[0]
        disease = label_encoder.inverse_transform([prediction])[0]
        confidence = round(max(probs) * 100, 2)

        st.success(f"🧾 Возможная болезнь: **{disease}**")
        st.info(f"📊 Уверенность модели: **{confidence}%**")

        # 🩻 Тяжесть симптомов
        severity_values = []
        for symptom in selected_symptoms:
            symptom_clean = symptom.strip().lower()
            row = severity_df[severity_df['Symptom'] == symptom_clean]
            if not row.empty:
                severity_values.append(int(row['weight'].values[0]))

        if severity_values:
            avg_severity = round(sum(severity_values) / len(severity_values), 2)
            st.markdown(f"### 🩻 Средняя тяжесть симптомов: **{avg_severity} / 5**")

            if avg_severity >= 4:
                st.error("⚠️ Высокая тяжесть симптомов — немедленно обратитесь к врачу!")
            elif avg_severity >= 2.5:
                st.warning("❗ Средняя тяжесть — желательно наблюдение.")
            else:
                st.success("✅ Симптомы выражены слабо.")
        else:
            st.info("Не удалось оценить тяжесть — проверь названия симптомов.")

        # 🩹 Рекомендации
        matching_row = precautions_df[precautions_df['Disease'].str.lower() == disease.lower()]
        if not matching_row.empty:
            st.markdown("### 🩹 Рекомендации:")
            for i in range(1, 5):
                advice = matching_row[f'Precaution_{i}'].values[0]
                if pd.notna(advice):
                    st.markdown(f"- {advice}")
        else:
            st.warning("⚠️ Рекомендации не найдены для этого заболевания.")

        # 📘 Описание болезни
        desc_row = desc_df[desc_df['Disease'].str.lower() == disease.lower()]
        if not desc_row.empty:
            description = desc_row['Description'].values[0]
            st.markdown("### 📘 Описание болезни:")
            st.markdown(f"{description}")
        else:
            st.info("Описание для этой болезни пока недоступно.")