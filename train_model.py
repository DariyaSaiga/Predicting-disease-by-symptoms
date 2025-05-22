import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Загрузка данных
df = pd.read_csv('data/dataset.csv')

# 2. Заполнение пропусков
df.fillna('None', inplace=True)

# 3. Получение списка всех уникальных симптомов
symptom_columns = [col for col in df.columns if 'Symptom_' in col]
all_symptoms = sorted({symptom for col in symptom_columns for symptom in df[col].unique() if symptom != 'None'})

# 4. Создание признаков: one-hot encoding симптомов
def encode_symptoms(row):
    present = [0] * len(all_symptoms)
    for symptom in row[symptom_columns]:
        if symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            present[idx] = 1
    return pd.Series(present)

X = df[symptom_columns].apply(encode_symptoms, axis=1)

# 5. Кодируем болезни (target)
le = LabelEncoder()
y = le.fit_transform(df['Disease'])

# 6. Обучение модели
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 7. Оценка
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 8. Сохранение модели и кодировщика
joblib.dump(clf, 'model/rf_model.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(all_symptoms, 'model/symptom_list.pkl')

print("✅ Модель обучена и сохранена.")