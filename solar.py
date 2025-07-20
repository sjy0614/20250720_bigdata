import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# ---------------------
# 📌 설정
st.set_page_config(page_title="태양광 유휴부지 예측", layout="wide")

st.title("☀️ 유휴부지 태양광 발전량 예측 대시보드")

# ---------------------
# 📂 데이터 불러오기
@st.cache_data
def load_data():
    df_train = pd.read_excel("최종_태양광.xlsx")
    df_idle = pd.read_excel("유휴부지_월별_날씨포함.xlsx")
    return df_train, df_idle

df_train, df_idle = load_data()

# ---------------------
# 🧪 전처리
df_train["연"] = df_train["날짜"].astype(str).str[:4].astype(int)
df_train["월"] = df_train["날짜"].astype(str).str[4:6].astype(int)
df_idle["연"] = df_idle["날짜"].astype(str).str[:4].astype(int)
df_idle["월"] = df_idle["날짜"].astype(str).str[4:6].astype(int)

features = [
    '설비용량(kW)', '위도', '경도', '연', '월',
    'NASA_평균기온(°C)', 'NASA_최고기온(°C)', 'NASA_최저기온(°C)',
    'NASA_풍속_10m(m/s)', 'NASA_풍속_50m(m/s)', 'NASA_강수량(mm)',
    'NASA_상대습도(%)', 'NASA_구름량(%)', 'NASA_투과도',
    'NASA_일사량(MJ/m²)', 'NASA_맑은하늘_일사량(MJ/m²)', 'NASA_지표면온도(°C)'
]

df_train = df_train.dropna(subset=features + ['발전량(kWh)'])

X = df_train[features]
y = df_train['발전량(kWh)']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------
# 👤 사용자 입력
st.sidebar.header("⚙️ 사용자 입력")
capacity_kw = st.sidebar.number_input("설비용량 (kW)", min_value=50, max_value=1000, step=50, value=300)
area_per_kw = st.sidebar.slider("1kW당 필요한 면적 (㎡)", 15, 30, 23)
min_area = capacity_kw * area_per_kw

# ---------------------
# 🔍 유휴부지 필터링
df_filtered = df_idle[df_idle["면적(m^2)"] >= min_area].copy()
df_filtered["설비용량(kW)"] = capacity_kw
df_filtered = df_filtered.dropna(subset=features)
X_pred = df_filtered[features]
df_filtered["예측_발전량(kWh)"] = model.predict(X_pred)

# ---------------------
# 📊 요약 정리
df_summary = (
    df_filtered.groupby("재산 소재지", as_index=False)
    .agg(
        위도=("위도", "first"),
        경도=("경도", "first"),
        면적=("면적(m^2)", "first"),
        연간_평균_예측_발전량_kWh=("예측_발전량(kWh)", "mean")
    )
)
df_summary = df_summary.sort_values("연간_평균_예측_발전량_kWh", ascending=False)

# ---------------------
# 🗺️ 지도 시각화
st.subheader("📍 유휴부지 위치 지도")

if not df_summary.empty:
    m = folium.Map(location=[df_summary["위도"].mean(), df_summary["경도"].mean()], zoom_start=7)
    for _, row in df_summary.iterrows():
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=7,
            popup=(f"{row['재산 소재지']}<br>예측 발전량: {row['연간_평균_예측_발전량_kWh']:.1f} kWh"),
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.6
        ).add_to(m)
    st_folium(m, width=1000, height=600)
else:
    st.warning("조건을 만족하는 유휴부지가 없습니다.")

# ---------------------
# 📋 표 출력
st.subheader("📊 유휴부지 예측 요약 (정렬순: 발전량)")
st.dataframe(df_summary.reset_index(drop=True))
