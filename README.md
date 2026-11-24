# 👨‍🍳 ChefBot — Asistente Culinario Multimodal en Telegram

Bot inteligente para **Telegram** desarrollado en Python que recomienda recetas personalizadas a partir de **búsqueda por texto (NLP)** o **reconocimiento de ingredientes mediante imágenes (Visión por Computadora)**.

---

## ✨ Características Principales

* **✍️ Búsqueda Inteligente por Texto (NLP):** Utiliza vectorización **TF-IDF** y **Similitud del Coseno** (`scikit-learn`) para encontrar la receta más compatible con los ingredientes escritos por el usuario en lenguaje natural.
* **📸 Reconocimiento Visual de Ingredientes:** Integra la API de **Google Gemini (`gemini-3.8-flash`)** para analizar fotografías tomadas desde Telegram, extraer los alimentos visibles y cruzarlos automáticamente con la base de datos local.
* **🎲 Exploración sin Repetición:** Permite filtrar recetas por categorías (*Desayunos, Almuerzos, Fit / Proteico*) o modo aleatorio manteniendo un historial por sesión para no repetir platos consecutivos.
* **🔄 Control de Flujo Interactivo:** Menú con botones inline y comandos dedicados (`/start`, `/nuevo`, `/reiniciar`, `/finalizar`, `/sorprendeme`).

---

## 🛠️ Stack Tecnológico

* **Lenguaje:** Python 3.11+
* **Bot API:** `pyTelegramBotAPI`
* **Procesamiento de Datos y NLP:** `pandas`, `scikit-learn` (TF-IDF + Cosine Similarity)
* **Inteligencia Artificial Multimodal:** `google-genai` (Gemini 3.8 Flash) y `Pillow`
* **Gestión de Configuración:** `python-dotenv`

---

## 📂 Estructura del Proyecto

```text
chefbot-telegram/
├── data/
│   └── recetas.csv        # Base de conocimiento de recetas, categorías e ingredientes
├── .env.example           # Plantilla de variables de entorno requeridas
├── .gitignore             # Exclusión de entorno virtual y credenciales (.env)
├── requirements.txt       # Dependencias del proyecto
├── chefbot.py             # Lógica principal del bot, NLP y visión por computadora
└── README.md              # Documentación del proyecto