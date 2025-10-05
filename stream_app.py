import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ------------------------------
# Chargement des données
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

df = load_data()

# ------------------------------
# Configuration de la page
# ------------------------------
st.set_page_config(page_title="Analyse des Frais Médicaux", page_icon="🏥", layout="wide")

st.title("🏥 Application Interactive : Analyse des Frais Médicaux")
st.write("""
Cette application explore les **facteurs qui influencent les frais médicaux** à partir du dataset *Insurance* de Kaggle.  
Elle combine **visualisation, interprétation automatique** et un **modèle prédictif simple** pour illustrer le lien entre
les variables démographiques, le mode de vie et les coûts de santé.
""")

# ------------------------------
# Navigation par onglets
# ------------------------------
tabs = st.tabs(["📊 Exploration", "📈 Modèle prédictif", "🧠 Insights automatiques"])

# ==========================================================
# 1️⃣ Onglet Exploration
# ==========================================================
with tabs[0]:
    st.header("📊 Exploration des variables")

    regions = ["Toutes les régions"] + sorted(df["region"].unique().tolist())
    region = st.selectbox("🌍 Sélectionnez une région :", regions)

    # Gestion du filtre global
    if region == "Toutes les régions":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["region"] == region]

    # --- Boxplot tabagisme
    st.subheader(f"🚬 Impact du tabagisme sur les frais médicaux ({region})")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=filtered_df, x="smoker", y="charges", palette="coolwarm", ax=ax1)
    st.pyplot(fig1)

    median_smoker = filtered_df[filtered_df["smoker"]=="yes"]["charges"].median()
    median_non = filtered_df[filtered_df["smoker"]=="no"]["charges"].median()
    ratio = median_smoker / median_non if median_non > 0 else 0

    st.markdown(f"""
    💬 **Observation :**  
    - Médiane fumeurs : **{median_smoker:,.0f} €**  
    - Médiane non-fumeurs : **{median_non:,.0f} €**  
    👉 Les fumeurs paient environ **{ratio:.1f}× plus** en frais médicaux.
    """)

    # --- Corrélation âge / frais
    st.subheader("🎂 Relation entre l’âge et les frais médicaux")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=filtered_df, x="age", y="charges", hue="smoker", alpha=0.7, palette="coolwarm", ax=ax2)
    st.pyplot(fig2)

    cor_age = filtered_df["age"].corr(filtered_df["charges"])
    st.markdown(f"""
    💬 **Analyse :**  
    Corrélation âge/frais : **{cor_age:.2f}**  
    👉 Les frais augmentent avec l’âge, surtout chez les fumeurs.
    """)

    # --- Corrélation BMI / frais
    st.subheader("⚖️ Relation entre le BMI et les frais médicaux")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=filtered_df, x="bmi", y="charges", hue="smoker", alpha=0.7, palette="coolwarm", ax=ax3)
    st.pyplot(fig3)

    cor_bmi = filtered_df["bmi"].corr(filtered_df["charges"])
    st.markdown(f"""
    💬 **Analyse :**  
    Corrélation BMI/frais : **{cor_bmi:.2f}**  
    👉 Un BMI élevé (>30) tend à augmenter les coûts, mais le **tabagisme reste le facteur dominant**.
    """)

# ==========================================================
# 2️⃣ Onglet Modèle prédictif
# ==========================================================
with tabs[1]:
    st.header("📈 Modèle de Régression Linéaire")

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("charges", axis=1)
    y = df_encoded["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    st.write("### 🧮 Entrez les paramètres pour estimer les frais médicaux :")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Âge :", 18, 64, 30)
        children = st.selectbox("Nombre d'enfants :", [0, 1, 2, 3, 4, 5])
    with col2:
        bmi = st.slider("BMI :", 15.0, 50.0, 25.0)
        smoker = st.selectbox("Fumeur :", ["yes", "no"])
    with col3:
        sex = st.selectbox("Sexe :", ["male", "female"])
        region_input = st.selectbox("Région :", sorted(df["region"].unique().tolist()))

    sample = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region_input == "northwest" else 0],
        "region_southeast": [1 if region_input == "southeast" else 0],
        "region_southwest": [1 if region_input == "southwest" else 0]
    })

    prediction = model.predict(sample)[0]
    st.success(f"💰 **Estimation des frais médicaux : {prediction:,.2f} €**")

    # Commentaires dynamiques
    if smoker == "yes":
        st.info("🚭 Le statut de fumeur augmente fortement les coûts médicaux.")
    if bmi > 30:
        st.warning("⚠️ Un BMI supérieur à 30 accroît significativement les dépenses médicales.")
    if age > 50:
        st.info("📈 L’âge avancé est associé à une hausse des frais médicaux moyens.")

    st.caption("🔧 Modèle linéaire en cours de développement — à des fins éducatives.")
# Section : Historique du Dataset
st.sidebar.markdown("### 🗂️ Historique du Dataset")
if st.sidebar.checkbox("Afficher l'historique des données"):
    st.markdown("""
    ### 📘 Historique du Dataset - *Insurance Charges (Kaggle)*  
    Le dataset **Insurance** provient de la plateforme [Kaggle](https://www.kaggle.com/).  
    Il contient des informations sur les **frais médicaux individuels** en fonction de variables démographiques et comportementales :
    
    - 👤 **age** : âge du bénéficiaire de l’assurance  
    - ⚖️ **bmi** : indice de masse corporelle  
    - 🧒 **children** : nombre d’enfants à charge  
    - 🚬 **smoker** : indique si la personne fume  
    - 🌍 **region** : région de résidence  
    - 💰 **charges** : frais médicaux facturés  

    **Objectif** : comprendre et modéliser les facteurs influençant le coût des soins de santé afin d’optimiser la tarification des assurances.  
    """)



# ==========================================================
# 3️⃣ Onglet Insights automatiques
# ==========================================================
with tabs[2]:
    st.header("🧠 Synthèse automatique des insights")

    st.markdown("""
    Cette section génère une **interprétation automatique** des tendances observées dans les données.  
    Idéale pour le **data storytelling** et la **présentation à la direction**.
    """)

    st.markdown("### 📋 Résumé global :")
    st.write(f"- **Corrélation âge/frais :** {cor_age:.2f}")
    st.write(f"- **Corrélation BMI/frais :** {cor_bmi:.2f}")
    st.write(f"- **Impact du tabagisme :** environ {ratio:.1f}× plus de dépenses pour les fumeurs.")
    st.write("- **Différences régionales :** faibles variations, tendance générale similaire.")

    st.markdown("---")
    st.subheader("🧩 Interprétation globale :")

    interpretation = f"""
    > Le **tabagisme** demeure le facteur dominant des coûts de santé, amplifiant les dépenses d’un facteur 3 à 4.  
    > Le **BMI** et l’**âge** jouent un rôle secondaire mais significatif dans l’augmentation des frais.  
    > Globalement, les **tendances régionales restent cohérentes**, ce qui montre que les effets sont 
    davantage liés au comportement qu’à la localisation.  
    > Ces résultats soutiennent des politiques de **prévention santé** et d’**ajustement du risque assurantiel**.
    """

    st.markdown(interpretation)

    st.success("✅ Interprétation automatique générée à partir des tendances du dataset.")
    st.caption("Analyse réalisée par **Kossi Noumagno — Data Analyst | Machine Learning & Data Storytelling**")
