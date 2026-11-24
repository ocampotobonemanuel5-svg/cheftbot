import os
import io
import telebot
from telebot import types
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Cargar variables de entorno
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

bot = telebot.TeleBot(TOKEN)
ai_client = genai.Client(api_key=GEMINI_KEY)

# 2. Cargar base de datos y entrenar modelo NLP (TF-IDF)
df = pd.read_csv("data/recetas.csv")
df["caracteristicas"] = df["nombre"] + " " + df["categoria"] + " " + df["ingredientes"]

vectorizador = TfidfVectorizer()
matriz_tfidf = vectorizador.fit_transform(df["caracteristicas"])

# Historial por chat para evitar repetir recetas consecutivas
historial_vistos = {}

def obtener_receta_sin_repetir(chat_id, sub_df):
    vistos = historial_vistos.get(chat_id, [])
    disponibles = sub_df[~sub_df.index.isin(vistos)]
    
    # Si ya vio todas las recetas de esa categoría, reinicia su historial
    if disponibles.empty:
        historial_vistos[chat_id] = []
        disponibles = sub_df
        
    elegida = disponibles.sample(1).iloc[0]
    historial_vistos.setdefault(chat_id, []).append(elegida.name)
    return elegida

def buscar_en_dataset(consulta_texto):
    vector_usuario = vectorizador.transform([consulta_texto])
    similitudes = cosine_similarity(vector_usuario, matriz_tfidf).flatten()
    indice_mejor = similitudes.argmax()
    
    if similitudes[indice_mejor] < 0.15:
        return None
    return df.iloc[indice_mejor]

def formatear_receta(receta):
    ingredientes_lista = "\n".join([f"  • {ing.strip().capitalize()}" for ing in receta["ingredientes"].split(",")])
    pasos = receta["preparacion"].replace("\\n", "\n")
    
    return (
        f"🍽️ *{receta['nombre'].upper()}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📂 *Categoría:* {receta['categoria']}  |  ⏱️ *Tiempo:* {receta['tiempo']}  |  👥 *Porciones:* {receta['porciones']}\n\n"
        f"🛒 *INGREDIENTES:*\n{ingredientes_lista}\n\n"
        f"🔥 *PREPARACIÓN:*\n{pasos}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )

def menu_principal():
    markup = types.InlineKeyboardMarkup(row_width=2)
    # Modos de entrada
    btn_texto = types.InlineKeyboardButton("✍️ Escribir ingredientes", callback_data="modo_texto")
    btn_foto = types.InlineKeyboardButton("📸 Tomar/Enviar foto", callback_data="modo_foto")
    # Categorías
    btn_desayuno = types.InlineKeyboardButton("🍳 Desayunos", callback_data="cat_Desayunos")
    btn_almuerzo = types.InlineKeyboardButton("🍛 Almuerzos", callback_data="cat_Almuerzos")
    btn_fit = types.InlineKeyboardButton("💪 Fit / Proteico", callback_data="cat_Fit")
    btn_random = types.InlineKeyboardButton("🎲 Sorpréndeme", callback_data="random")
    # Reinicio y finalización de flujo
    btn_reiniciar = types.InlineKeyboardButton("🔄 Nueva consulta / Finalizar", callback_data="reiniciar")
    
    markup.add(btn_texto, btn_foto)
    markup.add(btn_desayuno, btn_almuerzo, btn_fit, btn_random)
    markup.add(btn_reiniciar)
    return markup

# 3. Comandos de inicio, nueva búsqueda, reinicio y finalización
@bot.message_handler(commands=['start'])
def comando_start(message):
    texto = (
        "👨‍🍳 *¡Bienvenido a ChefBot!*\n\n"
        "¿Qué ingredientes tienes hoy o qué quieres preparar?\n\n"
        "1️⃣ *Por texto:* Escríbeme los ingredientes que tienes.\n"
        "2️⃣ *Por cámara:* Envíame una *foto* de los ingredientes y los identificaré.\n\n"
        "O explora directamente por categoría:"
    )
    bot.send_message(message.chat.id, texto, parse_mode="Markdown", reply_markup=menu_principal())

@bot.message_handler(commands=['nuevo'])
def comando_nuevo(message):
    texto = (
        "🆕 *Nueva búsqueda lista*\n\n"
        "Dime qué nuevos ingredientes tienes en tu cocina o envíame una foto 📸:"
    )
    bot.send_message(message.chat.id, texto, parse_mode="Markdown", reply_markup=menu_principal())

@bot.message_handler(commands=['reiniciar'])
def comando_reiniciar(message):
    historial_vistos[message.chat.id] = []
    texto = (
        "🔄 *ChefBot reiniciado desde cero*\n\n"
        "Se ha restablecido tu historial de recetas. Escribe tus ingredientes o envíame una foto 📸:"
    )
    bot.send_message(message.chat.id, texto, parse_mode="Markdown", reply_markup=menu_principal())

@bot.message_handler(commands=['finalizar'])
def comando_finalizar(message):
    texto = (
        "✅ *Consulta finalizada. ¡Buen provecho!*\n\n"
        "Cuando quieras cocinar algo más, escribe nuevos ingredientes, envía una foto o usa /start:"
    )
    bot.send_message(message.chat.id, texto, parse_mode="Markdown", reply_markup=menu_principal())

@bot.message_handler(commands=['sorprendeme'])
def comando_sorprendeme(message):
    receta = obtener_receta_sin_repetir(message.chat.id, df)
    bot.send_message(message.chat.id, formatear_receta(receta), parse_mode="Markdown", reply_markup=menu_principal())

# 4. Manejador de botones interactivos
@bot.callback_query_handler(func=lambda call: True)
def manejar_botones(call):
    chat_id = call.message.chat.id
    
    if call.data == "reiniciar":
        bot.send_message(
            chat_id,
            "✅ *Consulta finalizada.*\n\n👨‍🍳 Dime, ¿qué nuevos ingredientes tienes o qué plato buscamos ahora? (Escríbelos o envía una foto 📸)",
            parse_mode="Markdown",
            reply_markup=menu_principal()
        )
    elif call.data == "modo_texto":
        bot.send_message(
            chat_id,
            "✍️ *Modo Texto activado*\nEscribe los ingredientes que tienes en tu cocina.\n_Ejemplo: tengo pollo, arroz y crema de leche_",
            parse_mode="Markdown"
        )
    elif call.data == "modo_foto":
        bot.send_message(
            chat_id,
            "📸 *Modo Visión activado*\nToma una foto con la cámara de Telegram (clip 📎 > Cámara) o sube una imagen de tus ingredientes.",
            parse_mode="Markdown"
        )
    elif call.data == "random":
        receta = obtener_receta_sin_repetir(chat_id, df)
        bot.send_message(chat_id, formatear_receta(receta), parse_mode="Markdown", reply_markup=menu_principal())
    elif call.data.startswith("cat_"):
        categoria = call.data.split("_")[1]
        sub_df = df[df["categoria"] == categoria]
        if not sub_df.empty:
            receta = obtener_receta_sin_repetir(chat_id, sub_df)
            bot.send_message(chat_id, formatear_receta(receta), parse_mode="Markdown", reply_markup=menu_principal())
            
    bot.answer_callback_query(call.id)

# 5. Manejador de FOTOS (Visión por Computadora + TF-IDF)
@bot.message_handler(content_types=['photo'])
def analizar_foto_ingredientes(message):
    msg_espera = bot.reply_to(message, "🔍 *Analizando los ingredientes de tu foto...*", parse_mode="Markdown")
    
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        imagen = Image.open(io.BytesIO(downloaded_file))
        
        prompt = (
            "Identifica los alimentos e ingredientes de cocina visibles en esta imagen. "
            "Responde ÚNICAMENTE con la lista de nombres de los ingredientes en español separados por comas, "
            "sin saludos ni explicaciones. Ejemplo: pechuga de pollo, arroz, cebolla, ajo. "
            "Si no hay comida en la imagen, responde exactamente: NINGUNO"
        )
        respuesta_ia = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[imagen, prompt]
        )
        
        ingredientes_detectados = respuesta_ia.text.strip()
        
        if "NINGUNO" in ingredientes_detectados.upper() or not ingredientes_detectados:
            bot.edit_message_text(
                "❌ No logré identificar alimentos en esa imagen. Intenta tomar una foto más clara de los ingredientes.",
                chat_id=message.chat.id,
                message_id=msg_espera.message_id,
                reply_markup=menu_principal()
            )
            return
        
        receta = buscar_en_dataset(ingredientes_detectados)
        
        if receta is not None:
            texto_resultado = (
                f"📸 *Ingredientes detectados:* _{ingredientes_detectados}_\n\n"
                f"{formatear_receta(receta)}"
            )
            bot.edit_message_text(
                texto_resultado,
                chat_id=message.chat.id,
                message_id=msg_espera.message_id,
                parse_mode="Markdown",
                reply_markup=menu_principal()
            )
        else:
            bot.edit_message_text(
                f"📸 *Detecté:* _{ingredientes_detectados}_\n\n"
                "🤔 Aún no tengo una receta en mi base de datos que combine esos ingredientes exactos. Prueba con otros o elige una categoría:",
                chat_id=message.chat.id,
                message_id=msg_espera.message_id,
                parse_mode="Markdown",
                reply_markup=menu_principal()
            )
            
    except Exception as e:
        bot.edit_message_text(
            f"⚠️ Ocurrió un error al procesar la imagen: {e}",
            chat_id=message.chat.id,
            message_id=msg_espera.message_id
        )

# 6. Manejador de TEXTO libre
@bot.message_handler(func=lambda message: True)
def buscar_por_texto(message):
    texto_limpio = message.text.lower().strip()
    
    if texto_limpio in ["gracias", "listo", "fin", "finalizar", "terminar", "nueva", "nuevo", "otra", "hola", "menu", "menú"]:
        comando_nuevo(message)
        return

    receta = buscar_en_dataset(message.text)
    if receta is None:
        bot.reply_to(
            message,
            "🤔 No encontré una receta exacta con esos ingredientes. Prueba escribiendo otros o envíame una foto:",
            reply_markup=menu_principal()
        )
    else:
        bot.send_message(message.chat.id, formatear_receta(receta), parse_mode="Markdown", reply_markup=menu_principal())

if __name__ == "__main__":
    print("🚀 ChefBot Multimodal (Texto + Visión Gemini 3.8 Flash) iniciado en Telegram...")
    bot.infinity_polling()