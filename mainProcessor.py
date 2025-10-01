import os
import sys
from thermalLogicFace import analisis_termico_rostro
from thermalLogicalHands import analisis_termico_manos

# =========================================================================
# --- CÓDIGO MAIN: PUNTO DE ENTRADA ---
# =========================================================================

if __name__ == "__main__":
    
    # Verificar que se pasó un argumento
    if len(sys.argv) < 2:
        print("Error: No se proporcionó la ruta de la imagen.")
        print("Uso: python mainProcessor.py <ruta_imagen>")
        sys.exit(1)
    
    # Obtener la ruta de la imagen desde los argumentos
    IMAGEN_ENTRADA = sys.argv[1]
    
    if not os.path.exists(IMAGEN_ENTRADA):
        print(f"Error: La imagen de entrada no se encuentra en la ruta: {IMAGEN_ENTRADA}")
        sys.exit(1)

    # --- Cálculo de Rutas de Salida (Se guardarán junto a la imagen de entrada) ---
    BASE_DIR = os.path.dirname(IMAGEN_ENTRADA)
    if not BASE_DIR: 
        BASE_DIR = "." 

    # Generar nombres de archivos de salida basados en el nombre de la imagen
    imagen_nombre_base = os.path.splitext(os.path.basename(IMAGEN_ENTRADA))[0]
    
    # --- Rutas de SALIDA para el ROSTRO ---
    IMAGEN_ETIQUETADA_TEMP_ROSTRO = os.path.join(BASE_DIR, imagen_nombre_base + "_rostro_etiquetado.png")
    PDF_SALIDA_ROSTRO = os.path.join(BASE_DIR, imagen_nombre_base + "_Informe_Rostro.pdf")
    
    # --- Rutas de SALIDA para las MANOS ---
    IMAGEN_ETIQUETADA_TEMP_MANOS = os.path.join(BASE_DIR, imagen_nombre_base + "_manos_etiquetada.png")
    PDF_SALIDA_MANOS = os.path.join(BASE_DIR, imagen_nombre_base + "_Informe_Manos.pdf")

    print("--- INICIO DEL PROCESO DE ANÁLISIS INTEGRADO (ROSTRO Y MANOS) ---")
    print(f"Procesando imagen: {IMAGEN_ENTRADA}")
    print(f"Directorio de salida: {BASE_DIR}")
    
    # 1. --- PROCESO PARA EL ROSTRO ---
    print("\n==============================================")
    print("      INICIANDO ANÁLISIS DE ROSTRO 👤")
    print("==============================================")
    
    try:
        analisis_termico_rostro(
            image_path=IMAGEN_ENTRADA, 
            pdf_filename=PDF_SALIDA_ROSTRO, 
            imagen_etiquetada_path=IMAGEN_ETIQUETADA_TEMP_ROSTRO
        )
        print(f"✅ Rostro procesado. Imagen: {IMAGEN_ETIQUETADA_TEMP_ROSTRO}")
    except Exception as e:
        print(f"❌ Error en análisis de rostro: {e}")
        sys.exit(1)
    
    # 2. --- PROCESO PARA LAS MANOS ---
    print("\n==============================================")
    print("      INICIANDO ANÁLISIS DE MANOS 👋")
    print("==============================================")

    try:
        analisis_termico_manos(
            image_path=IMAGEN_ENTRADA, 
            pdf_filename=PDF_SALIDA_MANOS, 
            imagen_etiquetada_path=IMAGEN_ETIQUETADA_TEMP_MANOS
        )
        print(f"✅ Manos procesadas. Imagen: {IMAGEN_ETIQUETADA_TEMP_MANOS}")
    except Exception as e:
        print(f"❌ Error en análisis de manos: {e}")
        sys.exit(1)
    
    print("\n--- PROCESO INTEGRADO FINALIZADO ---")
    
    # Verificar que los archivos se crearon correctamente
    archivos_esperados = [
        IMAGEN_ETIQUETADA_TEMP_ROSTRO,
        PDF_SALIDA_ROSTRO,
        IMAGEN_ETIQUETADA_TEMP_MANOS,
        PDF_SALIDA_MANOS
    ]
    
    print("\n--- VERIFICACIÓN DE ARCHIVOS GENERADOS ---")
    for archivo in archivos_esperados:
        if os.path.exists(archivo):
            print(f"✅ {os.path.basename(archivo)}")
        else:
            print(f"❌ NO ENCONTRADO: {os.path.basename(archivo)}")