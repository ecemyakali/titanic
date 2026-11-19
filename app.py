import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Passenger Insights", layout="wide")
st.title("Titanic Passenger Insights")
st.write(
    "Yolcu hayatta kalma örüntülerini ve veri hazırlama adımlarını incelemek için "
    "etkileşimli keşif paneli."
)

# 1. Veriyi Yükle
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(URL)

selected_cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
df_work = df[selected_cols].copy()

# 2. Load and Inspect
st.header("1. Veri Yükleme ve İlk İnceleme")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Temel Bilgiler")
    st.write(f"**Satır sayısı:** {df_work.shape[0]}")
    st.write(f"**Sütun sayısı:** {df_work.shape[1]}")
    st.write("**Sütun isimleri:**", list(df_work.columns))
    st.write("**Veri tipleri:**")
    st.dataframe(df_work.dtypes.astype(str).rename("dtype"))

with col2:
    st.subheader("İlk 5 Satır ve Eksik Değerler")
    st.write("**İlk 5 satır (`head()`):**")
    st.dataframe(df_work.head())
    st.write("**Eksik değer sayıları (Ham veri):**")
    st.dataframe(df_work.isnull().sum().rename("missing_count"))

# .loc ve .iloc kullanım örneği
st.subheader(".loc ve .iloc ile Seçim Örneği")
st.write("İlk 3 satır ve `pclass`, `sex`, `age` sütunları (`.loc`):")
st.dataframe(df_work.loc[0:2, ["pclass", "sex", "age"]])
st.write("İlk 3 satır ve ilk 4 sütun (`.iloc`):")
st.dataframe(df_work.iloc[0:3, 0:4])

# 3. Handle Missing Values
st.header("2. Eksik Değerlerin Yönetimi")
st.info("Not: `deck` sütunu çok fazla eksik veri içerdiği için analiz dışında bırakılmıştır.")

missing_before = df_work.isnull().sum()

age_median = df_work["age"].median()
embarked_mode = df_work["embarked"].mode()[0]

df_clean = df_work.copy()
df_clean["age"] = df_clean["age"].fillna(age_median)
df_clean["embarked"] = df_clean["embarked"].fillna(embarked_mode)

missing_after = df_clean.isnull().sum()

miss_comp = pd.DataFrame({
    "Eksik (Önce)": missing_before,
    "Eksik (Sonra)": missing_after,
})
st.dataframe(miss_comp)
st.caption(
    f"`age` eksik değerleri medyan ({age_median:.1f}) ile, "
    f"`embarked` eksik değerleri en sık kategori ('{embarked_mode}') ile dolduruldu."
)

# 4. Create Transformed Features
st.header("3. Dönüştürülmüş Özellikler (Normalizasyon ve Yaş Grubu)")

# Min-Max normalizasyon (orijinal age ve fare sütunları korunur)
age_min, age_max = df_clean["age"].min(), df_clean["age"].max()
fare_min, fare_max = df_clean["fare"].min(), df_clean["fare"].max()

df_clean["age_minmax"] = (df_clean["age"] - age_min) / (age_max - age_min)
df_clean["fare_minmax"] = (df_clean["fare"] - fare_min) / (fare_max - fare_min)

# Yaş grubu ayrıklaştırma
bins = [-1, 12, 18, 35, 60, float("inf")]
labels = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
df_clean["age_group"] = pd.cut(df_clean["age"], bins=bins, labels=labels)

st.dataframe(
    df_clean[["age", "age_minmax", "age_group", "fare", "fare_minmax"]].head(10)
)

# 5. Apply Dummy Coding
st.header("4. Dummy Coding (`sex` ve `embarked`)")
st.write("Dummy coding öncesi önizleme (`sex`, `embarked`, `survived`):")
st.dataframe(df_clean[["sex", "embarked", "survived"]].head())

# survived sütununa dummy uygulanmaz; dummy sütunlar normalize edilmez
dummies = pd.get_dummies(
    df_clean[["sex", "embarked"]],
    prefix=["sex", "embarked"],
    dtype=int,
)
df_encoded = pd.concat([df_clean, dummies], axis=1)

st.write("Dummy coding sonrası önizleme:")
st.dataframe(df_encoded[["sex", "embarked"] + list(dummies.columns) + ["survived"]].head())

# 6. Interactive Dashboard
st.header("5. Etkileşimli Yolcu Analizi Paneli")

st.sidebar.header("Filtreler")
pclass_options = sorted(df_clean["pclass"].dropna().unique().tolist())
sex_options = sorted(df_clean["sex"].dropna().unique().tolist())
age_group_options = [g for g in labels if g in df_clean["age_group"].dropna().unique()]

selected_pclass = st.sidebar.multiselect(
    "Yolcu Sınıfı (pclass)", options=pclass_options, default=pclass_options
)
selected_sex = st.sidebar.multiselect(
    "Cinsiyet (sex)", options=sex_options, default=sex_options
)
selected_age_groups = st.sidebar.multiselect(
    "Yaş Grubu (age_group)", options=age_group_options, default=age_group_options
)

filtered_df = df_clean[
    df_clean["pclass"].isin(selected_pclass)
    & df_clean["sex"].isin(selected_sex)
    & df_clean["age_group"].isin(selected_age_groups)
]

m1, m2, m3, m4 = st.columns(4)
total_passengers = len(filtered_df)
survived_count = int(filtered_df["survived"].sum()) if total_passengers > 0 else 0
survival_rate = (survived_count / total_passengers * 100) if total_passengers > 0 else 0.0
avg_fare = filtered_df["fare"].mean() if total_passengers > 0 else 0.0

m1.metric("Filtrelenen Yolcu", total_passengers)
m2.metric("Hayatta Kalan", survived_count)
m3.metric("Hayatta Kalma Oranı", f"%{survival_rate:.1f}")
m4.metric("Ortalama Ücret (Fare)", f"{avg_fare:.2f}")

if total_passengers > 0:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sınıfa Göre Hayatta Kalma Oranı")
        class_surv = filtered_df.groupby("pclass", observed=False)["survived"].mean()
        st.bar_chart(class_surv)

    with c2:
        st.subheader("Yaş Grubuna Göre Yolcu Sayısı")
        age_counts = filtered_df["age_group"].value_counts().reindex(labels).fillna(0)
        st.bar_chart(age_counts)
else:
    st.warning("Seçili filtrelere uyan kayıt bulunamadı. Lütfen filtreleri genişletin.")

# 7. Write One Finding
st.header("6. Gözlemlenen Örüntü / Bulgu")
st.write(
    "Filtrelenen yolcular incelendiğinde, yolcu sınıfı (`pclass`) ve cinsiyet (`sex`) ile "
    "hayatta kalma oranı (`survived`) arasında belirgin bir birliktelik görülmektedir; özellikle "
    "1. sınıftaki ve kadın yolcuların hayatta kalma oranları, 3. sınıftaki yolculara kıyasla "
    "daha yüksek seyretmektedir. Bu gözlem değişkenler arasındaki ilişkiyi gösterir ve tek "
    "başına nedensellik anlamına gelmez."
)