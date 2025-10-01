import os
# Importamos la función principal del módulo de lógica de rostro
from thermalLogicFace import analisis_termico_rostro
# Importamos la NUEVA función del módulo de lógica de manos
from thermalLogicalHands import analisis_termico_manos # <-- NUEVA IMPORTACIÓN

# =========================================================================
# --- CÓDIGO MAIN: PUNTO DE ENTRADA ---
# =========================================================================

if __name__ == "__main__":
    
    # 🚨 ÚNICA RUTA QUE DEBES MODIFICAR 🚨
    # Asegúrate de que esta ruta sea válida y que los archivos DNN estén en el mismo directorio.
    IMAGEN_ENTRADA = "/Users/ceugeniogr/Library/Mobile Documents/com~apple~CloudDocs/Pagina+Servidor/carpetaFinal/FLIR3305.jpg"
    
    if not os.path.exists(IMAGEN_ENTRADA):
        print(f"Error: La imagen de entrada no se encuentra en la ruta: {IMAGEN_ENTRADA}")
        exit()

    # --- Cálculo de Rutas de Salida (Se guardarán junto a la imagen de entrada) ---
    BASE_DIR = os.path.dirname(IMAGEN_ENTRADA)
    if not BASE_DIR: 
        BASE_DIR = "." 

    # Generar nombres de archivos de salida basados en el nombre de la imagen
    imagen_nombre_base = os.path.splitext(os.path.basename(IMAGEN_ENTRADA))[0]
    
    # --- Rutas de SALIDA para el ROSTRO ---
    IMAGEN_ETIQUETADA_TEMP_ROSTRO = os.path.join(BASE_DIR, imagen_nombre_base + "_rostro_etiquetado.png") # Se guarda con el nombre base ahora
    PDF_SALIDA_ROSTRO = os.path.join(BASE_DIR, imagen_nombre_base + "_Informe_Rostro.pdf")
    
    # --- Rutas de SALIDA para las MANOS (NUEVAS RUTAS) ---
    IMAGEN_ETIQUETADA_TEMP_MANOS = os.path.join(BASE_DIR, imagen_nombre_base + "_manos_etiquetada.png")
    PDF_SALIDA_MANOS = os.path.join(BASE_DIR, imagen_nombre_base + "_Informe_Manos.pdf")
    # ----------------------------------------------------------------------------

    print("--- INICIO DEL PROCESO DE ANÁLISIS INTEGRADO (ROSTRO Y MANOS) ---")
    
    # 1. --- PROCESO PARA EL ROSTRO ---
    print("\n==============================================")
    print("      INICIANDO ANÁLISIS DE ROSTRO 👤")
    print("==============================================")
    
    # Llama a la función de rostro
    analisis_termico_rostro(
        image_path=IMAGEN_ENTRADA, 
        pdf_filename=PDF_SALIDA_ROSTRO, 
        imagen_etiquetada_path=IMAGEN_ETIQUETADA_TEMP_ROSTRO
    )
    
    # 2. --- PROCESO PARA LAS MANOS ---
    print("\n==============================================")
    print("      INICIANDO ANÁLISIS DE MANOS 👋")
    print("==============================================")

    # Llama a la NUEVA función de manos
    analisis_termico_manos(
        image_path=IMAGEN_ENTRADA, 
        pdf_filename=PDF_SALIDA_MANOS, 
        imagen_etiquetada_path=IMAGEN_ETIQUETADA_TEMP_MANOS
    )
    
    print("\n--- PROCESO INTEGRADO FINALIZADO ---")