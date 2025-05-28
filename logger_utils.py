# logger_utils.py - v1.6.5
import logging
import sys

# Determina il livello di logging (può essere configurato da config.yaml o secrets in futuro)
LOG_LEVEL_STR = "INFO" # Default a INFO
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR.upper(), logging.INFO)

def setup_logger(name: str, level: int = LOG_LEVEL) -> logging.Logger:
    """
    Configura e restituisce un logger.
    """
    # Evita di aggiungere handler multipli se il logger è già stato configurato
    # (utile se questa funzione viene chiamata più volte per lo stesso nome di logger)
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # Se ha già handler, assumiamo sia configurato e restituiamo
        # Potremmo voler controllare se il livello è quello desiderato e aggiustarlo,
        # ma per ora lo lasciamo così per semplicità.
        logger.setLevel(level) # Assicura che il livello sia quello passato
        return logger

    logger.setLevel(level)

    # Formatter per i log
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (%(module)s.%(funcName)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler per scrivere i log su stdout (che Streamlit Cloud cattura)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(stream_handler)
    
    # Opzionale: impedire la propagazione ai logger root se si vuole un controllo fine
    # logger.propagate = False 

    if name == "__main__": # Logger speciale per app.py
        print(f"INFO [logger_utils]: Logger principale '{name}' configurato con livello {logging.getLevelName(logger.level)}.")
    else:
        print(f"INFO [logger_utils]: Logger '{name}' configurato con livello {logging.getLevelName(logger.level)}.")


    return logger

if __name__ == '__main__':
    # Esempio di come usare il logger
    # In altri file: from logger_utils import setup_logger; logger = setup_logger(__name__)
    
    test_logger = setup_logger("TestLogger", level=logging.DEBUG)
    test_logger.debug("Questo è un messaggio di debug dal test logger.")
    test_logger.info("Questo è un messaggio informativo.")
    test_logger.warning("Questo è un avvertimento.")
    test_logger.error("Questo è un errore.")
    test_logger.critical("Questo è un errore critico.")

    # Esempio di logger per un modulo specifico
    module_logger = setup_logger("MyModule") # Userà il LOG_LEVEL di default (INFO)
    module_logger.debug("Questo messaggio debug da MyModule non verrà mostrato (perché il livello è INFO).")
    module_logger.info("Questo messaggio info da MyModule verrà mostrato.")
