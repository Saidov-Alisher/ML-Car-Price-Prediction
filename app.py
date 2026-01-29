import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Настройки страницы
st.set_page_config(
    page_title="🚗 Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("🚗 Прогноз цены автомобиля")
st.markdown("### Используется модель RandomForest для предсказания цены на основе характеристик")

# Загрузка модели
@st.cache_resource
def load_model():
    model_path = './models/rf_car_price.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"❌ Модель не найдена: {model_path}")
        return None

model = load_model()

# Загрузка данных для статистики
@st.cache_data
def load_dataset():
    csv_path = './CarPrice_Assignment.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df = load_dataset()

# Боковая панель для ввода
st.sidebar.header("📋 Параметры автомобиля")

# Числовые признаки
wheelbase = st.sidebar.slider("Колёсная база (см)", 86.0, 120.0, 95.0)
carlength = st.sidebar.slider("Длина (см)", 140.0, 210.0, 170.0)
carwidth = st.sidebar.slider("Ширина (см)", 60.0, 72.0, 66.0)
carheight = st.sidebar.slider("Высота (см)", 47.0, 60.0, 54.0)
curbweight = st.sidebar.slider("Вес (кг)", 1500, 4500, 2500)
enginesize = st.sidebar.slider("Объём двигателя (см³)", 68, 326, 150)
horsepower = st.sidebar.slider("Мощность (л.с.)", 46, 288, 100)
peakrpm = st.sidebar.slider("Макс. обороты (об/мин)", 4150, 6600, 5000)
citympg = st.sidebar.slider("Расход в городе (МПГ)", 15, 50, 25)
highwaympg = st.sidebar.slider("Расход на трассе (МПГ)", 16, 54, 30)

# Категориальные признаки
st.sidebar.subheader("🏷️ Категории")
brand = st.sidebar.selectbox(
    "Бренд",
    ['toyota', 'nissan', 'mazda', 'honda', 'mitsubishi', 'subaru', 
     'audi', 'bmw', 'volkswagen', 'porsche', 'volvo', 'dodge']
)

carbody = st.sidebar.selectbox(
    "Тип кузова",
    ['sedan', 'hatchback', 'wagon', 'convertible', 'hardtop']
)

drivewheel = st.sidebar.selectbox(
    "Привод",
    ['fwd', 'rwd', '4wd']
)

enginetype = st.sidebar.selectbox(
    "Тип двигателя",
    ['ohc', 'ohcv', 'l', 'dohc', 'rotor']
)

cylindernumber = st.sidebar.selectbox(
    "Количество цилиндров",
    ['three', 'four', 'five', 'six', 'eight', 'twelve']
)

fuelsystem = st.sidebar.selectbox(
    "Топливная система",
    ['mpfi', 'spdi', '2bbl', 'idi', 'mfi']
)

# Подготовка данных для предсказания
input_data = pd.DataFrame({
    'wheelbase': [wheelbase],
    'carlength': [carlength],
    'carwidth': [carwidth],
    'carheight': [carheight],
    'curbweight': [curbweight],
    'enginesize': [enginesize],
    'horsepower': [horsepower],
    'peakrpm': [peakrpm],
    'citympg': [citympg],
    'highwaympg': [highwaympg],
    'brand': [brand],
    'carbody': [carbody],
    'drivewheel': [drivewheel],
    'enginetype': [enginetype],
    'cylindernumber': [cylindernumber],
    'fuelsystem': [fuelsystem]
})

# Основной контент
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Параметры автомобиля")
    st.dataframe(input_data.iloc[:, :10].T, use_container_width=True)
    
    st.subheader("🏷️ Характеристики")
    st.dataframe(input_data.iloc[:, 10:].T, use_container_width=True)

with col2:
    st.subheader("💰 Прогноз цены")
    
    if model is not None:
        try:
            prediction = model.predict(input_data)[0]
            
            # Стиль вывода
            st.metric(
                label="Прогнозируемая цена",
                value=f"${prediction:,.2f}",
                delta=None
            )
            
            # Доверительный интервал (примерный)
            lower_bound = prediction * 0.85
            upper_bound = prediction * 1.15
            
            st.info(f"""
            📈 **Ожидаемый диапазон цены:**
            - Минимум: ${lower_bound:,.2f}
            - Максимум: ${upper_bound:,.2f}
            """)
            
        except Exception as e:
            st.error(f"❌ Ошибка при предсказании: {e}")
    else:
        st.error("❌ Модель не загружена")

# Таб со статистикой
st.divider()
st.subheader("📊 Статистика датасета")

if df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Основные данные", "🏆 Топ бренды", "🎨 Графики", "ℹ️ Информация"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего авто", len(df))
        col2.metric("Средняя цена", f"${df['price'].mean():,.0f}")
        col3.metric("Мин. цена", f"${df['price'].min():,.0f}")
        col4.metric("Макс. цена", f"${df['price'].max():,.0f}")
        
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        # Топ бренды
        if 'CarName' in df.columns:
            df_temp = df.copy()
            df_temp['brand'] = df_temp['CarName'].apply(lambda x: str(x).split()[0].lower())
            top_brands = df_temp.groupby('brand')['price'].agg(['count', 'mean']).sort_values('count', ascending=False).head(10)
            st.dataframe(top_brands.rename(columns={'count': 'Количество', 'mean': 'Средняя цена'}), use_container_width=True)
    
    with tab3:
        # Графики
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('./figures/correlation_matrix.png'):
                st.image('./figures/correlation_matrix.png', caption='Матрица корреляций')
            if os.path.exists('./figures/mean_price_top_brands.png'):
                st.image('./figures/mean_price_top_brands.png', caption='Топ бренды по цене')
        
        with col2:
            if os.path.exists('./figures/boxplot_price_top_brands.png'):
                st.image('./figures/boxplot_price_top_brands.png', caption='Распределение цен')
            if os.path.exists('./figures/feature_importances.png'):
                st.image('./figures/feature_importances.png', caption='Важность признаков')
    
    with tab4:
        st.write("**Описание датасета:**")
        st.dataframe(df.describe(), use_container_width=True)

st.divider()
st.caption("🚗 ML Car Price Prediction | RandomForest Model | Accuracy: R² = 0.958")
