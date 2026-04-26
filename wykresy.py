import pandas as pd
import matplotlib.pyplot as plt
import os

# Przygotowanie folderu na wykresy
output_dir = 'wykresy'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Polskie nazwy atrybutów
feature_names_pl = [
    "Dzieci z niepełnosprawnością", "Koszty projektów wodnych", "Rezolucja budżetowa", 
    "Zamrożenie opłat lekarzy", "Pomoc dla Salwadoru", "Religia w szkołach", 
    "Testy antysatelitarne", "Pomoc dla Contras", "Pociski MX", 
    "Imigracja", "Paliwa syntetyczne", "Wydatki na edukację", 
    "Prawo do pozwów (Superfund)", "Przestępczość", "Eksport bezcłowy", "Eksport do RPA"
]

# Wczytanie danych przez Pandas
try:
    df = pd.read_csv('house-votes-84.data', header=None)
    
    # Czyszczenie kolumny z partią
    df[0] = df[0].str.extract(r'(democrat|republican)', expand=False)
    
    # Przypisanie nazw kolumn
    df.columns = ['partia'] + feature_names_pl
except Exception as e:
    print(f"Błąd podczas wczytywania pliku: {e}")
    exit()

# Wykres rozkładu klas (Demokraci vs Republikanie)
try:
    party_order = ['democrat', 'republican']
    party_counts = df['partia'].value_counts().reindex(party_order).fillna(0).astype(int)
    party_pct = (party_counts / party_counts.sum()) * 100

    plt.figure(figsize=(5, 4.5))
    bars = plt.bar(
        ['Demokraci', 'Republikanie'],
        party_pct.values,
        width=0.45,
        color=['#00008B', '#8B0000'],
        edgecolor='black'
    )

    for bar, pct, cnt in zip(bars, party_pct.values, party_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{pct:.1f}%\n(n={cnt})",
            ha='center',
            va='bottom',
            fontsize=11
        )

    plt.ylabel('Udział rekordów (%)')
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig(f'{output_dir}/rozklad_klas.png', bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Błąd podczas generowania wykresu rozkładu klas: {e}")

# Kolory
colors = ['#00008B', '#4682B4', '#ADD8E6', '#8B0000', '#CD5C5C', '#FFB6C1']

# Pętla generująca osobne wykresy procentowe
for col in feature_names_pl:
    # Filtrowanie grup
    dem = df[df['partia'] == 'democrat'][col]
    rep = df[df['partia'] == 'republican'][col]
    
    # Obliczanie procentów (%) dla każdej partii osobno
    plot_data = pd.Series({
        'Demokraci (Tak)': (dem == 'y').mean() * 100,
        'Demokraci (Nie)': (dem == 'n').mean() * 100,
        'Demokraci (?)': (dem == '?').mean() * 100,
        'Republikanie (Tak)': (rep == 'y').mean() * 100,
        'Republikanie (Nie)': (rep == 'n').mean() * 100,
        'Republikanie (?)': (rep == '?').mean() * 100
    })

    # Tworzenie rysunku
    plt.figure(figsize=(10, 6))
    ax = plot_data.plot(kind='bar', color=colors, edgecolor='black')
    
    # Dodanie etykiet procentowych nad słupkami
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points', fontsize=10)

    # Estetyka
    plt.ylabel('Procent głosów wewnątrz partii (%)')
    plt.ylim(0, 115)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Zapis do folderu
    safe_name = col.replace(" ", "_").lower()
    plt.savefig(f'{output_dir}/{safe_name}.png', bbox_inches='tight')
    plt.close()

print(f"Gotowe! Wykresy zostały zapisane w folderze '{output_dir}'.")