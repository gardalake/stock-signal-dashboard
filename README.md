# AI-Powered Stock & Crypto Signal Dashboard (v1.6.5)

Questo progetto è un dashboard interattivo costruito con Streamlit per visualizzare segnali di trading per azioni e criptovalute. Utilizza dati da Alpha Vantage (per azioni) e CoinGecko (per criptovalute), applica un modello di machine learning (Random Forest o integrazione con Google AI Studio) per generare predizioni, e deriva segnali di trading (BUY/SELL/HOLD) e breakout.

## Caratteristiche Principali (basate sulle Specifiche v1.6.5)

*   **Frontend Interattivo:** Interfaccia utente costruita con Streamlit.
*   **Fonti Dati Multiple:**
    *   Azioni: Alpha Vantage
    *   Criptovalute: CoinGecko
*   **Modellazione ML:**
    *   Feature tecniche: MA20, MA50, RSI, StochRSI, Momentum (implementazione iniziale placeholder/semplice).
    *   Modello: RandomForestRegressor (allenato su dati storici).
    *   Predizioni: Variazione percentuale del prezzo a vari orizzonti temporali (focus su 3 giorni per i segnali, come da `config.yaml`).
    *   Opzione per integrare Google AI Studio per predizioni avanzate (placeholder).
*   **Logica dei Segnali:**
    *   Segnali BUY/SELL/HOLD basati sulle predizioni del modello.
    *   Rilevamento di Breakout (rialzisti e ribassisti) basati su prezzo e volume.
    *   Considerazione (teorica) degli spread di mercato.
*   **Visualizzazione:**
    *   Grafici a candela interattivi con Plotly.
    *   Visualizzazione di medie mobili, segnali ML e segnali di breakout sul grafico.
*   **Notifiche (Placeholder/Future):**
    *   Segnali audio per eventi di BUY/SELL (implementazione placeholder).
    *   Notifiche email opzionali via SMTP (implementazione placeholder).
*   **Configurabilità:** Parametri chiave gestiti tramite un file `config.yaml`.

## Struttura del Progetto

Il progetto è organizzato nei seguenti moduli principali:

*   `app.py`: Script principale dell'applicazione Streamlit e logica UI.
*   `data_utils.py`: Funzioni per il recupero e la pre-elaborazione dei dati da API esterne.
*   `ml_model.py`: Logica per il calcolo delle feature, training del modello ML e generazione delle predizioni.
*   `signal_logic.py`: Algoritmi per derivare segnali di trading, rilevare breakout e applicare logiche di business.
*   `visualization.py`: Funzioni per creare i grafici e le visualizzazioni dati con Plotly.
*   `sound_utils.py`: Gestione (placeholder) dei segnali audio.
*   `config.yaml`: File di configurazione per i parametri dell'applicazione.
*   `requirements.txt`: Dipendenze Python del progetto.
*   `.streamlit/secrets.toml`: **(Locale/Streamlit Cloud)** Per memorizzare chiavi API e altre credenziali sensibili. **Non commettere questo file su repository pubblici se contiene chiavi reali.**
*   `README.md`: Questo file.
*   `sounds/`: Cartella (opzionale) per i file audio `.wav`.

## Installazione e Esecuzione Locale

1.  **Clona il Repository:**
    ```bash
    git clone https://github.com/gardalake/stock-signal-dashboard.git
    cd stock-signal-dashboard
    ```

2.  **Crea un Ambiente Virtuale (Consigliato):**
    ```bash
    python -m venv venv
    # Su Linux/macOS:
    source venv/bin/activate
    # Su Windows:
    # venv\Scripts\activate
    ```

3.  **Installa le Dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configura i Secrets:**
    *   Crea una cartella `.streamlit` nella root del progetto (se non esiste già).
    *   All'interno di `.streamlit`, crea un file `secrets.toml`.
    *   Aggiungi la tua chiave API Alpha Vantage e altre credenziali necessarie:
        ```toml
        # Esempio per .streamlit/secrets.toml
        ALPHA_VANTAGE_API_KEY = "LA_TUA_CHIAVE_API_ALPHA_VANTAGE"
        
        # Opzionale, per Google AI Studio (come da config.yaml)
        # GOOGLE_AI_STUDIO_URL = "URL_ENDPOINT_AI_STUDIO"
        # GOOGLE_AI_STUDIO_TOKEN = "TOKEN_ENDPOINT_AI_STUDIO"

        # Opzionale, per notifiche email (come da config.yaml, se implementato)
        # EMAIL_SMTP_PASSWORD = "PASSWORD_SMTP"
        ```

5.  **Esegui l'Applicazione Streamlit:**
    ```bash
    streamlit run app.py
    ```
    L'applicazione dovrebbe aprirsi nel tuo browser web.

## Deployment su Streamlit Cloud

1.  **Fork e Clona (se necessario) o usa il tuo repository esistente.**
2.  **Assicurati che il repository su GitHub sia aggiornato** con tutti i file del progetto, inclusi `requirements.txt` e `config.yaml`.
3.  **Accedi a Streamlit Community Cloud** ([share.streamlit.io](https://share.streamlit.io/)).
4.  **Crea una nuova app ("New app"):**
    *   Seleziona il tuo repository GitHub e il branch corretto (solitamente `main` o `master`).
    *   Il file principale dell'app è `app.py`.
5.  **Configura i Secrets nell'interfaccia di Streamlit Cloud:**
    *   Vai nelle impostazioni dell'app ("Settings" o "Manage app").
    *   Nella sezione "Secrets", aggiungi le tue chiavi API (es. `ALPHA_VANTAGE_API_KEY`) e altre credenziali, replicando la struttura che avresti in `secrets.toml`. Non è necessario caricare il file `secrets.toml` stesso se il repo è pubblico.
6.  **Avvia il Deploy.** Streamlit Cloud installerà le dipendenze da `requirements.txt` e avvierà l'app.

## Troubleshooting

*   **Pagina Bianca/Nessun Output:**
    *   Controlla i log dell'app su Streamlit Cloud. Sono la fonte principale per errori Python.
    *   Assicurati che `app.py` contenga comandi di output di Streamlit (es. `st.title()`, `st.write()`, `st.dataframe()`, ecc.).
    *   Verifica che non ci siano loop infiniti o blocchi nel codice (specialmente nelle chiamate API o nei calcoli lunghi).
    *   Controlla la console del browser per errori JavaScript o WebSocket, anche se spesso gli errori Python nei log di Streamlit Cloud sono più indicativi.
*   **Errori di Importazione Moduli:**
    *   Assicurati che tutti i file `.py` del progetto siano nella stessa directory principale o in sottocartelle correttamente strutturate come package Python (con file `__init__.py` se necessario, anche se per importazioni dirette nella root non servono).
    *   Verifica che i nomi dei file e delle funzioni importate corrispondano esattamente.
*   **Errori API (Alpha Vantage, CoinGecko):**
    *   Verifica la validità della tua chiave API.
    *   Controlla i limiti di rate dell'API (Alpha Vantage ha limiti stringenti sul piano gratuito).
    *   I messaggi di errore delle API sono spesso stampati nei log o direttamente nell'interfaccia Streamlit (grazie ai messaggi `st.error` che abbiamo aggiunto).

## Roadmap (Prossimi Passi Basati su Specifiche v1.7.0)

*   Migliorare la robustezza e l'implementazione delle feature tecniche (es. usando la libreria `ta`).
*   Implementare pienamente i segnali audio (esplorare workarounds per Streamlit Cloud).
*   Sviluppare un bot Telegram per le notifiche.
*   Integrare Google AI Studio per l'ottimizzazione degli iperparametri e/o per sostituire il RandomForest locale.
*   Aggiungere grafici multi-timeframe.
*   Migliorare la gestione degli errori e il logging.

---
*Questo README è basato sulle specifiche del progetto "Stock_Signal_Dashboard_v1.6.5_Spec.txt".*
