import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from knowledge import ProductKnowledge
from datetime import datetime, timedelta

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
STAFF_ROLE_ID = int(os.getenv("STAFF_ROLE_ID"))
TICKET_CATEGORY_ID = int(os.getenv("TICKET_CATEGORY_ID"))

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)
knowledge = ProductKnowledge()

last_staff_reply = {}

# Palabras que indican solo un saludo
SALUDOS = {"hola", "buenas", "buen día", "buenos días", "buenas tardes", "buenas noches", 
           "hi", "hey", "hello", "saludos", "qué tal", "como estás", "cómo estás", "oe", "wena"}

# Palabras que indican que quiere info de productos
PALABRAS_PRODUCTO = {"precio", "precios", "cuanto", "cuánto", "cuesta", "costo", "info", 
                     "información", "plan", "planes", "producto", "productos", "basico", "básico", 
                     "extreme", "vip", "comprar", "quiero", "dame", "tienes", "tienen", "disponible",
                     "servicio", "servicios", "cheat", "hack", "menu", "menú", "funciones", "características",
                     "catalogo", "catálogo", "lista", "todos", "ofrecen", "cuales", "cuáles"}

def es_solo_saludo(texto: str) -> bool:
    texto_lower = texto.lower().strip()
    if len(texto_lower) < 15:
        for saludo in SALUDOS:
            if saludo in texto_lower:
                return True
    return False

def tiene_intencion_producto(texto: str) -> bool:
    texto_lower = texto.lower()
    for palabra in PALABRAS_PRODUCTO:
        if palabra in texto_lower:
            return True
    return False

def is_ticket_channel(channel: discord.TextChannel) -> bool:
    if channel.category_id == TICKET_CATEGORY_ID:
        return True
    if "ticket" in channel.name.lower():
        return True
    return False

@bot.event
async def on_ready():
    print(f'✅ Bot activo como {bot.user}')
    print(f'🧠 Productos cargados: {len(knowledge.documents)} documentos')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if message.author.bot:
        return
    if not message.content or message.content.strip() == "":
        await bot.process_commands(message)
        return
    if not isinstance(message.channel, discord.TextChannel):
        await bot.process_commands(message)
        return
    if not is_ticket_channel(message.channel):
        await bot.process_commands(message)
        return

    guild = message.guild
    member = message.author
    channel = message.channel
    staff_role = guild.get_role(STAFF_ROLE_ID)

    if staff_role is None:
        print(f"⚠️  No se encontró el rol de staff con ID {STAFF_ROLE_ID}")
        await bot.process_commands(message)
        return

    if staff_role in member.roles:
        last_staff_reply[channel.id] = datetime.now()
        await bot.process_commands(message)
        return

    last_reply = last_staff_reply.get(channel.id)
    if last_reply and datetime.now() - last_reply < timedelta(minutes=2):
        await bot.process_commands(message)
        return

    texto = message.content.strip()

    # ─── Si es SOLO un saludo ───
    if es_solo_saludo(texto):
        embed = discord.Embed(
            description=f"👋 ¡Hola {member.mention}! Bienvenido a nuestro soporte.\n\n¿En qué puedo ayudarte hoy? Puedes preguntarme sobre nuestros **planes**, **precios** o **funciones** 🎮",
            color=discord.Color.green()
        )
        await channel.send(embed=embed)
        await bot.process_commands(message)
        return

    # ─── Si NO menciona nada de productos ───
    if not tiene_intencion_producto(texto):
        embed = discord.Embed(
            description=f"🤔 {member.mention}, no estoy seguro de qué necesitas. ¿Puedes darme más detalles?\n\nTe puedo ayudar con:\n• 💰 **Precios** de nuestros planes\n• 📋 **Funciones** incluidas\n• 🎮 **Información** sobre productos",
            color=discord.Color.blue()
        )
        await channel.send(embed=embed)
        await bot.process_commands(message)
        return

    # ─── Si es una pregunta sobre productos ───
    async with channel.typing():
        result = knowledge.ask(texto)
        answer = result["answer"]
        confidence = result.get("confidence", 0)

        # Si el knowledge base no tiene productos cargados
        if "Aún no tengo productos cargados" in answer:
            embed = discord.Embed(
                title="⚠️ Sin productos cargados",
                description=f"{member.mention}, no tengo información de productos disponible en este momento. Un asesor te atenderá pronto.",
                color=discord.Color.red()
            )
            await channel.send(f"{staff_role.mention}", embed=embed)
            await bot.process_commands(message)
            return

        # Si necesita escalación (producto no encontrado o precio inventado)
        needs_human = any(phrase in answer.lower() for phrase in [
            "necesito consultar", "no tengo información", "no sé", 
            "no encuentro", "contacta con", "human", "equipo"
        ])

        if needs_human:
            embed = discord.Embed(
                title="👤 Escalando a un asesor humano",
                description=answer,
                color=discord.Color.orange()
            )
            if confidence > 0:
                embed.set_footer(text=f"Confianza del match: {confidence}")
            await channel.send(f"{staff_role.mention}", embed=embed)
        else:
            embed = discord.Embed(
                title="🛍️ Atención al Cliente",
                description=answer,
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            embed.set_footer(text=f"Respondiendo a {member.display_name} | Confianza: {confidence}")
            await channel.send(embed=embed)

    await bot.process_commands(message)

def has_staff_role():
    async def predicate(ctx):
        role = discord.utils.get(ctx.guild.roles, id=STAFF_ROLE_ID)
        if role is None or role not in ctx.author.roles:
            await ctx.send("❌ Necesitas ser staff para usar este comando.", delete_after=5)
            return False
        return True
    return commands.check(predicate)

@bot.command()
@has_staff_role()
async def recargar(ctx):
    knowledge._load_knowledge()
    await ctx.send(f"✅ Base de productos recargada. {len(knowledge.documents)} documentos activos.")

@bot.command()
async def productos(ctx):
    import glob
    files = glob.glob("products/*.txt")
    if not files:
        await ctx.send("📭 No hay productos cargados.")
        return
    names = [os.path.basename(f) for f in files]
    await ctx.send("📦 Productos en memoria:\n" + "\n".join(f"• `{n}`" for n in names))

bot.run(TOKEN)