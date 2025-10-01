# 1. Imagen Base de Python
FROM python:3.9-slim

# 2. Instalar Dependencias del Sistema (Tesseract y Librerías de OpenCV)
# Combinamos ambas necesidades para asegurar compatibilidad total
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libgthread-2.0-0

# 3. Directorio de Trabajo
WORKDIR /app

# 4. Copiar e Instalar Dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el Código de la Aplicación
COPY . .

# 6. Exponer Puerto para Render
EXPOSE 10000

# 7. Comando de Inicio para Flask con Gunicorn
# Apuntamos a server.py (el archivo) y app (la variable Flask)
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:10000", "server:app"]
