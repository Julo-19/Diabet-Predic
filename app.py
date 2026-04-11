import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Pour des graphiques plus esthétiques


st.set_page_config(page_title="Diabète AI-Predict Dashboard", layout="wide", page_icon="🩺")

# Style CSS 
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3.5em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        font-size: 1.1em;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
    }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Chargement des fichiers (Modèle, Scaler ET Données)
@st.cache_resource # Pour éviter de recharger à chaque clic
def load_assets():
    model = joblib.load('modele_diabete.pkl')
    scaler = joblib.load('scaler_diabete.pkl')
    # Chargement du dataset pour les graphiques comparatifs
    # Assure-toi que 'diabetes.csv' est dans le même dossier
    df_historical = pd.read_csv('diabetes.csv')
    return model, scaler, df_historical

try:
    model, scaler, df_historical = load_assets()
except FileNotFoundError:
    st.error("🚨 Erreur : Fichiers requis introuvables. Assurez-vous que 'modele_diabete.pkl', 'scaler_diabete.pkl' et 'diabetes.csv' sont dans le même dossier.")
    st.stop()

# Barre latérale (Sidebar) - Simplifiée
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864448.png", width=100)
    st.header("Informations Système")
    st.markdown(f"""
    **Modèle :** XGBoost Classifieur  
    **Base d'entraînement :** {len(df_historical)} patients (Pima)  
    **Précision :** ~77%
    """)
    st.divider()
    st.caption("⚠️ *Cet outil est une aide au diagnostic indicatif et ne remplace pas une consultation médicale.*")


st.title(" Système Intelligent de Prédiction du Risque de Diabète")
st.markdown("Utilisez le formulaire ci-dessous pour évaluer le risque de diabète basé sur l'IA.")
st.divider()

# Formulaire Patient 
st.subheader("Paramètres Biométriques du Patient")
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    with col1:
        st.markdown("####  Antécédents & Âge")
        preg = st.number_input("Nombre de Grossesses", 0, 20, 1, help="Nombre de fois que la patiente a été enceinte.")
        age = st.number_input("Âge (années)", 1, 120, 30)
        dpf = st.number_input("Score Hérédité (Pedigree)", 0.0, 3.0, 0.47, step=0.01, help="Fonction pedigree du diabète (score de risque génétique).")

    with col2:
        st.markdown("####  Analyses Sanguines")
        gluco = st.number_input("Glucose (mg/dL)", 0, 300, 120, help="Taux de glucose plasmatique à 2 heures lors d'un test d'hyperglycémie provoquée par voie orale.")
        insu = st.number_input("Insuline (mu U/ml)", 0, 900, 80, help="Taux d'insuline sérique à 2 heures (mu U/ml).")

    with col3:
        st.markdown("####  Mesures Corporelles")
        bmi = st.number_input("IMC (BMI)", 0.0, 70.0, 25.0, help="Indice de Masse Corporelle (Poids en kg / (Taille en m)^2).")
        bp = st.number_input("Pression Artérielle (mm Hg)", 0, 150, 70, help="Pression artérielle diastolique (mm Hg).")
        skin = st.number_input("Épaisseur Pli Cutané (mm)", 0, 100, 20, help="Épaisseur du pli cutané du triceps (mm).")

st.markdown("---")

#Logique de prédiction et affichage dynamique des résultats
if st.button(" LANCER L'ANALYSE DU RISQUE"):
    # Préparation des données pour le modèle
    features = np.array([[preg, gluco, bp, skin, insu, bmi, dpf, age]])
    # Application du scaling
    features_scaled = scaler.transform(features)
    
    # Prédiction et probabilité
    prediction = model.predict(features_scaled)
    proba = model.predict_proba(features_scaled)[0][1]

    # --- Section Résultats Visuels ---
    st.header(" Résultats de l'Analyse")
    
  
    main_res_col1, main_res_col2 = st.columns([1.2, 2], gap="large")

    # Colonne Gauche : Diagnostic & Interprétation
    with main_res_col1:
        st.subheader("📢 Diagnostic Suggéré")
        if prediction[0] == 1:
            st.error(f"🚨 RISQUE ÉLEVÉ DÉTECTÉ (Probabilité : {proba:.1%})")
            st.markdown("👉 *Le modèle suggère fortement que ce patient présente des marqueurs associés au diabète.*")
        else:
            st.success(f"✅ RISQUE FAIBLE (Probabilité : {proba:.1%})")
            st.markdown("👉 *Le modèle suggère que ce patient est sain.*")
        
        # Affichage de l'indicateur de confiance
        st.metric(label="Score de Confiance de l'IA", value=f"{proba:.1%}", help="Probabilité que le patient soit diabétique selon le modèle.")
        st.divider()

        # Interprétation détaillée
        st.subheader("🔍 Interprétation des Facteurs Clés")
        with st.expander("Voir l'analyse des facteurs influents", expanded=True):
            if gluco > 140:
                st.warning(f"🚩 **Alerte Glucose ({gluco} mg/dL) :** Votre taux est élevé. Le glucose est le facteur le plus influent pour le modèle.")
            else:
                st.info(f"✅ **Glucose ({gluco} mg/dL) :** Votre taux est dans la norme.")

            if bmi > 30:
                st.warning(f"🚩 **Alerte IMC ({bmi:.1f}) :** L'obésité (IMC > 30) augmente significativement le risque selon l'IA.")
            
            if age > 45:
                st.info(f"ℹ️ **Facteur Âge ({age} ans) :** Le risque augmente statistiquement avec l'âge.")
            
            if prediction[0] == 1:
                st.markdown("**Recommandation :** Une consultation médicale pour des examens complémentaires (ex: HbA1c) est vivement conseillée.")

    # Colonne Droite : Le Graphique Comparative Dynamique
    with main_res_col2:
        st.subheader("📊 Comparaison : Où se situe le patient ?")
        st.markdown(f"Ce graphique positionne le taux de **Glucose ({gluco} mg/dL)** de votre patient (ligne rouge) par rapport à la distribution historique des 768 patients du dataset Pima.")
        
        # Création du graphique Matplotlib/Seaborn
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Histogramme de fond (Population)
        sns.histplot(df_historical['Glucose'], bins=30, kde=True, color='#a0cfff', edgecolor='white', alpha=0.8, ax=ax)
        
        # Ligne verticale rouge pour le patient actuel
        ax.axvline(gluco, color='#dc3545', linestyle='--', linewidth=3, label='Ce Patient')
        
        # Personnalisation du graphique
        ax.set_title("Distribution du Taux de Glucose (Population vs Patient actuel)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Taux de Glucose (mg/dL)", fontsize=12)
        ax.set_ylabel("Nombre de Patients (Fréquence)", fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Amélioration du rendu visuel de Matplotlib
        sns.despine(left=True, bottom=True)
        
        # Affichage du graphique dans Streamlit
        st.pyplot(fig)
        st.caption("Source des données de fond : Pima Indians Diabetes Database (UCI Machine Learning Repository).")