# MLP_M — klasyfikacja afiliacji partyjnej na podstawie głosowań (MLP)

Projekt prezentuje implementację i serię eksperymentów z perceptronem wielowarstwowym (MLP) uczonym metodą wstecznej propagacji błędu z członem bezwładności (momentum). Celem jest klasyfikacja (binarna) na podstawie wektorów głosowań oraz porównanie wpływu architektury i hiperparametrów na skuteczność.

Obliczenia sieci (propagacja w przód, błędy, aktualizacje) są przyspieszone przez kompilację JIT (Numba), więc pierwsze uruchomienie może trwać dłużej ze względu na kompilację funkcji.

## Dane

Model uczy się na danych opisujących zachowania w głosowaniach (cechy) oraz etykiecie klasy (afiliacja partyjna). W projekcie przyjęto kodowanie:

- etykieta klasy: wartości binarne $\{-1, +1\}$
- głosowania: $y \rightarrow 1$, $n \rightarrow -1$, brak/niejednoznaczne dane $\rightarrow 0$

Dane robocze są przechowywane w postaci zserializowanej (format Hickle/HKL), co ułatwia szybkie uruchamianie eksperymentów.

## Model

Wykorzystywany MLP ma:

- dwie warstwy ukryte o rozmiarach $K_1$ i $K_2$
- wyjście jednowymiarowe dla klasyfikacji binarnej
- funkcję aktywacji typu tanh (tansig) w warstwach
- uczenie oparte o gradient + momentum
- metryki monitorowane podczas uczenia:
  - SSE (suma kwadratów błędów)
  - PK (procent poprawnych klasyfikacji, po progowaniu znaku wyjścia)

W eksperymentach stosowana jest walidacja krzyżowa typu Stratified K-Fold (domyślnie 10-fold), aby ograniczyć wpływ losowego podziału na wynik.

## Co przetestowano

W ramach projektu wykonano serię spójnych eksperymentów porównawczych:

- Wpływ architektury sieci: przeszukanie wartości $K_1$ i $K_2$ (rozmiary warstw ukrytych) i wybór konfiguracji dającej najwyższe PK w walidacji krzyżowej.
- Wpływ parametrów uczenia: przegląd siatki wartości współczynnika uczenia ($lr$) oraz momentum ($mc$) i wskazanie najlepszego kompromisu jakości.
- Ważność cech (głosowań): ocena wrażliwości modelu na poszczególne cechy poprzez „psucie” (zerowanie) pojedynczych wejść i pomiar spadku PK.
- Wpływ ilości danych: nauka na rosnącej liczbie próbek (przy stałym zbiorze testowym) i obserwacja, jak zmienia się skuteczność generalizacji.
- Grid Search 4D: pełne przeszukiwanie przestrzeni $\{K_1, K_2, lr, mc\}$ z zapisem wyników do pliku JSON oraz osobnym generowaniem wykresów z zapisanych rezultatów.

Wyniki eksperymentów są wizualizowane (wykresy 2D/3D, heatmapy, histogramy) i zapisywane do plików graficznych.

## Uruchamianie

### Wymagania

- Python 3.9+ (zalecane)
- Pakiety: `numpy`, `matplotlib`, `scikit-learn`, `hickle`, `numba`
- Dodatkowo do przygotowania danych: `pandas`

Instalacja zależności:

```bash
pip install numpy matplotlib scikit-learn hickle numba pandas
```

### Szybki start

1. Upewnij się, że w katalogu projektu znajduje się plik z danymi w formacie HKL (zserializowane dane wejściowe).
2. Uruchom wybrany skrypt eksperymentu (oznaczony prefiksem `exp`). Skrypty same wczytują dane, wykonują obliczenia i zapisują wykresy.

Wskazówka: eksperymenty z Grid Search mogą trwać długo; w projekcie przewidziano zapis wyników do JSON, aby później tylko odtwarzać wykresy bez ponownego liczenia.

### Przygotowanie danych (opcjonalnie)

Jeżeli zaczynasz od surowego pliku z głosowaniami:

1. Uruchom transformację do postaci numerycznej (mapowanie etykiet i wartości głosowań).
2. Zapisz wynik do formatu HKL, aby eksperymenty mogły wczytywać dane jednym krokiem.

## Reproduktywność i uwagi

- Walidacja krzyżowa jest stratyfikowana (zachowuje proporcje klas).
- W części eksperymentów stosowane jest losowanie podziałów (tam, gdzie to potrzebne); wyniki mogą minimalnie różnić się pomiędzy uruchomieniami.
- Dla porównywalności wyników kluczowe jest utrzymanie tych samych ustawień danych oraz zakresów przeszukiwań hiperparametrów.

## Raport

W repozytorium znajduje się również raport w LaTeX opisujący założenia, metodologię i wyniki w formie opisowej wraz z wykresami wygenerowanymi przez skrypty.
