## sound_utils.py - v1.6.5
import streamlit as st
import os # Per controllare l'esistenza dei file audio

# Questo modulo è un placeholder e la riproduzione audio automatica
# su Streamlit Cloud è problematica. 
# Le specifiche indicano questo come una feature desiderata.

def _attempt_play_sound(sound_file_path: str, play_sounds_enabled: bool):
    """
    Logica interna per tentare di riprodurre un file audio.
    Attualmente, logga solo l'azione o usa st.audio che mostra un player.
    L'autoplay è difficile da implementare in modo affidabile.
    """
    if not play_sounds_enabled:
        # st.info("[sound_utils]: Riproduzione suoni disabilitata nelle impostazioni.")
        return

    st.write(f"DEBUG [sound_utils]: Tentativo riproduzione suono: '{sound_file_path}'")

    # Verifica se il file audio esiste (percorso relativo alla root del progetto)
    if not os.path.exists(sound_file_path):
        st.warning(f"[sound_utils] ATTENZIONE: File audio '{sound_file_path}' non trovato. Assicurati che il percorso sia corretto e il file esista.")
        return

    try:
        # st.audio() mostra un widget player, non fa autoplay.
        # È il modo più semplice per "riprodurre" audio in Streamlit.
        # Per l'autoplay, servirebbero hack HTML/JS che sono sconsigliati o inaffidabili.
        # st.audio(sound_file_path, format='audio/wav', start_time=0) 
        
        # Dato che l'autoplay è complesso, per ora simuliamo con un messaggio.
        # In futuro, si potrebbe provare con st.components.v1.html e un tag <audio autoplay>,
        # ma i browser spesso lo bloccano a meno che non ci sia un'interazione utente.
        st.info(f"[sound_utils] SIMULAZIONE: Riproduzione suono '{os.path.basename(sound_file_path)}'. (Autoplay non implementato attivamente)")
        
    except Exception as e:
        st.error(f"[sound_utils] ERRORE durante il tentativo di gestione del suono '{sound_file_path}': {e}")


def play_buy_signal_sound(config_sound_utils: dict):
    """
    Chiama la funzione per riprodurre il suono del segnale di acquisto,
    se abilitato e il file è specificato in config.
    """
    play_enabled = config_sound_utils.get("play_sounds", False)
    sound_file = config_sound_utils.get("buy_sound_file")

    if play_enabled and sound_file:
        _attempt_play_sound(sound_file, play_enabled)
    elif play_enabled and not sound_file:
        st.warning("[sound_utils] ATTENZIONE: Riproduzione suono BUY abilitata, ma 'buy_sound_file' non specificato in config.")


def play_sell_signal_sound(config_sound_utils: dict):
    """
    Chiama la funzione per riprodurre il suono del segnale di vendita,
    se abilitato e il file è specificato in config.
    """
    play_enabled = config_sound_utils.get("play_sounds", False)
    sound_file = config_sound_utils.get("sell_sound_file")

    if play_enabled and sound_file:
        _attempt_play_sound(sound_file, play_enabled)
    elif play_enabled and not sound_file:
        st.warning("[sound_utils] ATTENZIONE: Riproduzione suono SELL abilitata, ma 'sell_sound_file' non specificato in config.")


if __name__ == '__main__':
    # Blocco per test standalone
    st.write("--- INIZIO TEST STANDALONE sound_utils.py ---")

    # Creare una configurazione di esempio
    sample_config_sounds_enabled = {
        "play_sounds": True,
        "buy_sound_file": "sounds/buy_signal_placeholder.wav",  # Assicurati che esista o crea un file fittizio
        "sell_sound_file": "sounds/sell_signal_placeholder.wav" # Assicurati che esista o crea un file fittizio
    }
    sample_config_sounds_disabled = {
        "play_sounds": False,
        "buy_sound_file": "sounds/buy_signal_placeholder.wav",
        "sell_sound_file": "sounds/sell_signal_placeholder.wav"
    }
    
    # Crea file audio fittizi se non esistono per il test
    sounds_dir = "sounds"
    if not os.path.exists(sounds_dir):
        os.makedirs(sounds_dir)
    
    placeholder_buy_sound = os.path.join(sounds_dir, "buy_signal_placeholder.wav")
    placeholder_sell_sound = os.path.join(sounds_dir, "sell_signal_placeholder.wav")

    if not os.path.exists(placeholder_buy_sound):
        with open(placeholder_buy_sound, "w") as f:
            f.write("dummy buy sound data") # File fittizio
    if not os.path.exists(placeholder_sell_sound):
        with open(placeholder_sell_sound, "w") as f:
            f.write("dummy sell sound data") # File fittizio
            
    st.write("\n--- Test con suoni ABILITATI ---")
    st.write("Test suono BUY:")
    play_buy_signal_sound(sample_config_sounds_enabled)
    st.write("Test suono SELL:")
    play_sell_signal_sound(sample_config_sounds_enabled)

    st.write("\n--- Test con suoni DISABILITATI ---")
    st.write("Test suono BUY (disabilitato):")
    play_buy_signal_sound(sample_config_sounds_disabled)
    st.write("Test suono SELL (disabilitato):")
    play_sell_signal_sound(sample_config_sounds_disabled)
    
    st.write("\n--- Test con file mancante (esempio) ---")
    config_missing_file = {
        "play_sounds": True,
        "buy_sound_file": "sounds/non_existent_sound.wav"
    }
    play_buy_signal_sound(config_missing_file)

    st.write("\n--- FINE TEST STANDALONE sound_utils.py ---")Audio signal support
