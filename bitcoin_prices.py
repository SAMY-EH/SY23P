import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings

# Supprimer les warnings pour un affichage plus propre
warnings.filterwarnings('ignore')

# Configuration de style pour les graphiques
plt.style.use('default')
sns.set_palette("husl")

def load_bitcoin_data(file_path):
    """
    Charge les données Bitcoin depuis le fichier CSV
    """
    try:
        # Charger les données
        df = pd.read_csv(file_path)
        
        # Convertir la colonne de temps en datetime
        df['Open time'] = pd.to_datetime(df['Open time'])
        
        # Nettoyer les noms de colonnes (enlever les espaces)
        df.columns = df.columns.str.strip()
        
        # Définir l'index comme datetime
        df.set_index('Open time', inplace=True)
        
        print(f"Données chargées avec succès!")
        print(f"Période: {df.index.min()} à {df.index.max()}")
        print(f"Nombre d'enregistrements: {len(df):,}")
        
        return df
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None

def basic_statistics(df):
    """
    Affiche les statistiques de base des données Bitcoin
    """
    print("\n" + "="*50)
    print("STATISTIQUES DESCRIPTIVES")
    print("="*50)
    
    # Statistiques pour les prix principaux
    price_cols = ['Open', 'High', 'Low', 'Close']
    print(df[price_cols].describe())
    
    print(f"\n📈 Prix le plus haut: ${df['High'].max():,.2f}")
    print(f"📉 Prix le plus bas: ${df['Low'].min():,.2f}")
    print(f"💰 Prix de clôture actuel: ${df['Close'].iloc[-1]:,.2f}")
    
    # Calcul de la volatilité quotidienne moyenne
    df['Daily_Range'] = ((df['High'] - df['Low']) / df['Open']) * 100
    print(f"🎯 Volatilité moyenne quotidienne: {df['Daily_Range'].mean():.2f}%")

def plot_price_evolution(df):
    """
    Graphique d'évolution des prix Bitcoin
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analyse des Prix Bitcoin (2018-2025)', fontsize=16, fontweight='bold')
    
    # 1. Évolution du prix de clôture
    axes[0, 0].plot(df.index, df['Close'], linewidth=1, color='orange')
    axes[0, 0].set_title('Évolution du Prix de Clôture')
    axes[0, 0].set_ylabel('Prix (USD)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Volume des transactions
    axes[0, 1].plot(df.index, df['Volume'], linewidth=1, color='blue', alpha=0.7)
    axes[0, 1].set_title('Volume des Transactions')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Candlestick simplifié (High-Low)
    axes[1, 0].fill_between(df.index, df['Low'], df['High'], alpha=0.3, color='green')
    axes[1, 0].plot(df.index, df['Close'], linewidth=1, color='red')
    axes[1, 0].set_title('Range High-Low avec Prix de Clôture')
    axes[1, 0].set_ylabel('Prix (USD)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Distribution des prix de clôture
    axes[1, 1].hist(df['Close'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_title('Distribution des Prix de Clôture')
    axes[1, 1].set_xlabel('Prix (USD)')
    axes[1, 1].set_ylabel('Fréquence')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_yearly_analysis(df):
    """
    Analyse par année
    """
    # Ajouter une colonne année
    df_copy = df.copy()
    df_copy['Year'] = df_copy.index.year
    
    # Calculer les statistiques par année
    yearly_stats = df_copy.groupby('Year').agg({
        'Close': ['mean', 'min', 'max'],
        'Volume': 'mean',
        'High': 'max',
        'Low': 'min'
    }).round(2)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analyse Annuelle Bitcoin', fontsize=16, fontweight='bold')
    
    # 1. Prix moyen par année
    yearly_avg = df_copy.groupby('Year')['Close'].mean()
    axes[0, 0].bar(yearly_avg.index, yearly_avg.values, color='gold', edgecolor='black')
    axes[0, 0].set_title('Prix Moyen par Année')
    axes[0, 0].set_ylabel('Prix Moyen (USD)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Volume moyen par année
    yearly_volume = df_copy.groupby('Year')['Volume'].mean()
    axes[0, 1].bar(yearly_volume.index, yearly_volume.values, color='lightblue', edgecolor='black')
    axes[0, 1].set_title('Volume Moyen par Année')
    axes[0, 1].set_ylabel('Volume Moyen')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Range (High - Low) par année
    df_copy['Range'] = df_copy['High'] - df_copy['Low']
    yearly_range = df_copy.groupby('Year')['Range'].mean()
    axes[1, 0].bar(yearly_range.index, yearly_range.values, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('Volatilité Moyenne par Année (High-Low)')
    axes[1, 0].set_ylabel('Range Moyen (USD)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Boxplot des prix par année
    years_to_plot = sorted(df_copy['Year'].unique())
    data_for_box = [df_copy[df_copy['Year'] == year]['Close'] for year in years_to_plot]
    axes[1, 1].boxplot(data_for_box, labels=years_to_plot)
    axes[1, 1].set_title('Distribution des Prix par Année')
    axes[1, 1].set_ylabel('Prix (USD)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("STATISTIQUES PAR ANNÉE")
    print("="*50)
    print(yearly_stats)

def plot_correlation_analysis(df):
    """
    Analyse des corrélations entre les différentes variables
    """
    # Sélectionner les colonnes numériques principales
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    correlation_data = df[numeric_cols]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Analyse des Corrélations', fontsize=16, fontweight='bold')
    
    # 1. Matrice de corrélation
    corr_matrix = correlation_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[0])
    axes[0].set_title('Matrice de Corrélation')
    
    # 2. Scatter plot Volume vs Prix
    axes[1].scatter(df['Volume'], df['Close'], alpha=0.5, s=1)
    axes[1].set_xlabel('Volume')
    axes[1].set_ylabel('Prix de Clôture (USD)')
    axes[1].set_title('Volume vs Prix de Clôture')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Fonction principale pour exécuter l'analyse
    """
    print("🚀 ANALYSE DES DONNÉES BITCOIN")
    print("="*50)
    
    # Charger les données
    file_path = "btc_15m_data_2018_to_2025.csv"
    df = load_bitcoin_data(file_path)
    
    if df is None:
        print("❌ Impossible de charger les données")
        return
    
    # Afficher les statistiques de base
    basic_statistics(df)
    
    # Créer les visualisations
    print("\n📊 Génération des graphiques...")
    
    # Évolution des prix
    plot_price_evolution(df)
    
    # Analyse par année
    plot_yearly_analysis(df)
    
    # Analyse des corrélations
    plot_correlation_analysis(df)
    
    print("\n✅ Analyse terminée!")

if __name__ == "__main__":
    main()