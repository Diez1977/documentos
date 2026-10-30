"""
RAG documental con Python, ChromaDB, Gemini Embeddings y Groq.

Archivo independiente del buscador anterior.
- No modifica la colección existente "buscador_documentos".
- Crea y utiliza una colección nueva: "documentos_rag".
- Lee PDF y DOCX desde RUTA_DOCUMENTOS.
- Permite indexar/actualizar documentos y hacer preguntas sobre ellos.

Variables esperadas en .env:
    GEMINI_API_KEY=...
    GROQ_API_KEY=...
    IA_MODELO=openai/gpt-oss-20b
    IA_SYSTEM=...
    RUTA_DOCUMENTOS=documentos

Dependencias:
    pip install chromadb google-genai groq python-dotenv pypdf python-docx

Nota:
    La primera indexación puede tardar porque genera un embedding por fragmento.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from docx import Document
from google import genai
from groq import Groq
from pypdf import PdfReader


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MODELO_EMBEDDING = os.getenv(
    "IA_EMBEDDING_MODELO",
    "gemini-embedding-2"
).strip()

MODELO_IA = os.getenv(
    "IA_MODELO",
    "openai/gpt-oss-20b"
).strip()

INSTRUCCIONES_SISTEMA = os.getenv(
    "IA_SYSTEM",
    "Eres un asistente que responde con precisión basándose en los documentos proporcionados."
).strip()

ruta_documentos_env = os.getenv("RUTA_DOCUMENTOS", "documentos").strip()
RUTA_DOCUMENTOS = Path(ruta_documentos_env)
if not RUTA_DOCUMENTOS.is_absolute():
    RUTA_DOCUMENTOS = BASE_DIR / RUTA_DOCUMENTOS

RUTA_CHROMA = BASE_DIR / "chroma_db"
NOMBRE_COLECCION = "documentos_rag"

TAMANO_CHUNK = 1000
SOLAPAMIENTO = 200
RESULTADOS_RECUPERACION = 5
MAX_CARACTERES_CONTEXTO = 18000
PAUSA_ENTRE_EMBEDDINGS = 1.0
MAX_REINTENTOS = 3

EXTENSIONES_ADMITIDAS = {".pdf", ".docx"}


# ============================================================
# CLIENTES Y COLECCIÓN
# ============================================================

def validar_configuracion() -> None:
    """Comprueba las claves necesarias antes de iniciar."""
    faltan = []

    if not GEMINI_API_KEY:
        faltan.append("GEMINI_API_KEY")

    if not GROQ_API_KEY:
        faltan.append("GROQ_API_KEY")

    if faltan:
        raise RuntimeError(
            "Faltan estas variables en el archivo .env: "
            + ", ".join(faltan)
        )


def crear_clientes() -> tuple[genai.Client, Groq]:
    validar_configuracion()
    cliente_gemini = genai.Client(api_key=GEMINI_API_KEY)
    cliente_groq = Groq(api_key=GROQ_API_KEY)
    return cliente_gemini, cliente_groq


def obtener_coleccion():
    """Abre la colección RAG, separada de la colección del buscador anterior."""
    cliente_chroma = chromadb.PersistentClient(path=str(RUTA_CHROMA))

    return cliente_chroma.get_or_create_collection(
        name=NOMBRE_COLECCION,
        metadata={"hnsw:space": "cosine"}
    )


# ============================================================
# LECTURA DE DOCUMENTOS
# ============================================================

def leer_pdf(ruta: Path) -> str:
    lector = PdfReader(str(ruta))
    paginas = []

    for numero, pagina in enumerate(lector.pages, start=1):
        texto = pagina.extract_text() or ""
        if texto.strip():
            paginas.append(f"[Página {numero}]\n{texto}")

    return "\n\n".join(paginas)


def leer_docx(ruta: Path) -> str:
    documento = Document(str(ruta))
    partes = []

    for parrafo in documento.paragraphs:
        texto = parrafo.text.strip()
        if texto:
            partes.append(texto)

    for tabla in documento.tables:
        for fila in tabla.rows:
            valores = [
                celda.text.strip().replace("\n", " ")
                for celda in fila.cells
            ]
            if any(valores):
                partes.append(" | ".join(valores))

    return "\n".join(partes)


def leer_documento(ruta: Path) -> str:
    extension = ruta.suffix.casefold()

    if extension == ".pdf":
        return leer_pdf(ruta)

    if extension == ".docx":
        return leer_docx(ruta)

    raise ValueError(f"Formato no admitido: {extension}")


def localizar_documentos() -> list[Path]:
    if not RUTA_DOCUMENTOS.exists():
        return []

    return sorted(
        ruta for ruta in RUTA_DOCUMENTOS.rglob("*")
        if ruta.is_file() and ruta.suffix.casefold() in EXTENSIONES_ADMITIDAS
    )


# ============================================================
# FRAGMENTACIÓN
# ============================================================

def dividir_en_fragmentos(
    texto: str,
    tamano: int = TAMANO_CHUNK,
    solapamiento: int = SOLAPAMIENTO
) -> list[str]:
    texto = " ".join(texto.split())

    if not texto:
        return []

    if tamano <= 0:
        raise ValueError("El tamaño del fragmento debe ser mayor que cero.")

    if solapamiento < 0 or solapamiento >= tamano:
        raise ValueError(
            "El solapamiento debe ser mayor o igual que cero "
            "y menor que el tamaño del fragmento."
        )

    paso = tamano - solapamiento
    fragmentos = []

    for inicio in range(0, len(texto), paso):
        fragmento = texto[inicio:inicio + tamano].strip()
        if fragmento:
            fragmentos.append(fragmento)

        if inicio + tamano >= len(texto):
            break

    return fragmentos


# ============================================================
# EMBEDDINGS
# ============================================================

def generar_embedding(
    cliente_gemini: genai.Client,
    texto: str
) -> list[float]:
    """Genera un vector con reintentos ante errores temporales."""
    ultimo_error: Exception | None = None

    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            respuesta = cliente_gemini.models.embed_content(
                model=MODELO_EMBEDDING,
                contents=texto
            )

            if not respuesta.embeddings:
                raise RuntimeError("Gemini no ha devuelto ningún embedding.")

            vector = respuesta.embeddings[0].values

            if not vector:
                raise RuntimeError("Gemini ha devuelto un embedding vacío.")

            return list(vector)

        except Exception as error:
            ultimo_error = error
            if intento < MAX_REINTENTOS:
                espera = intento * 2
                print(
                    f"  Error generando embedding. "
                    f"Reintento {intento}/{MAX_REINTENTOS - 1} "
                    f"en {espera} segundos..."
                )
                time.sleep(espera)

    raise RuntimeError(
        f"No se pudo generar el embedding: {ultimo_error}"
    )


# ============================================================
# INDEXACIÓN
# ============================================================

def crear_id_fragmento(nombre_documento: str, numero: int, texto: str) -> str:
    contenido = f"{nombre_documento}|{numero}|{texto}"
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()


def indexar_documento(
    ruta: Path,
    cliente_gemini: genai.Client,
    coleccion: Any
) -> int:
    nombre = ruta.name
    print(f"\nProcesando: {nombre}")

    texto = leer_documento(ruta)
    fragmentos = dividir_en_fragmentos(texto)

    if not fragmentos:
        print("  El documento no contiene texto extraíble. Se omite.")
        return 0

    # Eliminamos únicamente los fragmentos anteriores de este documento
    # dentro de la colección nueva de RAG.
    try:
        coleccion.delete(where={"documento_origen": nombre})
    except Exception:
        # Si todavía no existen fragmentos para este documento, continuamos.
        pass

    ids = []
    documentos = []
    metadatos = []
    embeddings = []

    for numero, fragmento in enumerate(fragmentos, start=1):
        print(
            f"  Generando embedding {numero}/{len(fragmentos)}...",
            end="\r",
            flush=True
        )

        vector = generar_embedding(cliente_gemini, fragmento)

        ids.append(crear_id_fragmento(nombre, numero, fragmento))
        documentos.append(fragmento)
        metadatos.append({
            "documento_origen": nombre,
            "numero_chunk": numero
        })
        embeddings.append(vector)

        if numero < len(fragmentos):
            time.sleep(PAUSA_ENTRE_EMBEDDINGS)

    coleccion.add(
        ids=ids,
        documents=documentos,
        metadatas=metadatos,
        embeddings=embeddings
    )

    print(
        f"  Indexación completada: {len(fragmentos)} fragmentos."
        + " " * 20
    )
    return len(fragmentos)


def indexar_documentos(
    cliente_gemini: genai.Client,
    coleccion: Any
) -> None:
    rutas = localizar_documentos()

    if not rutas:
        print(
            f"No se han encontrado archivos PDF o DOCX en:\n"
            f"{RUTA_DOCUMENTOS}"
        )
        return

    print(f"\nDocumentos encontrados: {len(rutas)}")
    total_fragmentos = 0
    errores = []

    for ruta in rutas:
        try:
            total_fragmentos += indexar_documento(
                ruta,
                cliente_gemini,
                coleccion
            )
        except Exception as error:
            errores.append((ruta.name, str(error)))
            print(f"\nError procesando {ruta.name}: {error}")

    print("\n" + "=" * 60)
    print("RESUMEN DE INDEXACIÓN")
    print("=" * 60)
    print(f"Documentos encontrados: {len(rutas)}")
    print(f"Fragmentos añadidos: {total_fragmentos}")
    print(f"Fragmentos totales en colección RAG: {coleccion.count()}")

    if errores:
        print(f"\nDocumentos con errores: {len(errores)}")
        for nombre, error in errores:
            print(f"- {nombre}: {error}")


# ============================================================
# RECUPERACIÓN DE CONTEXTO
# ============================================================

def recuperar_fragmentos(
    pregunta: str,
    cliente_gemini: genai.Client,
    coleccion: Any,
    cantidad: int = RESULTADOS_RECUPERACION
) -> list[dict[str, Any]]:
    if coleccion.count() == 0:
        return []

    vector_pregunta = generar_embedding(cliente_gemini, pregunta)

    resultados = coleccion.query(
        query_embeddings=[vector_pregunta],
        n_results=min(cantidad, coleccion.count()),
        include=["documents", "metadatas", "distances"]
    )

    documentos = resultados.get("documents", [[]])[0]
    metadatos = resultados.get("metadatas", [[]])[0]
    distancias = resultados.get("distances", [[]])[0]

    fragmentos = []

    for texto, metadata, distancia in zip(
        documentos,
        metadatos,
        distancias
    ):
        fragmentos.append({
            "texto": texto,
            "documento": metadata.get("documento_origen", "Desconocido"),
            "numero_chunk": metadata.get("numero_chunk", "?"),
            "distancia": float(distancia)
        })

    return fragmentos


def construir_contexto(
    fragmentos: list[dict[str, Any]],
    max_caracteres: int = MAX_CARACTERES_CONTEXTO
) -> str:
    partes = []
    caracteres = 0

    for fragmento in fragmentos:
        cabecera = (
            f"[Fuente: {fragmento['documento']} | "
            f"Fragmento: {fragmento['numero_chunk']}]\n"
        )
        bloque = cabecera + fragmento["texto"] + "\n\n"

        espacio_restante = max_caracteres - caracteres
        if espacio_restante <= 0:
            break

        if len(bloque) > espacio_restante:
            bloque = bloque[:espacio_restante]

        partes.append(bloque)
        caracteres += len(bloque)

    return "\n".join(partes)


# ============================================================
# GENERACIÓN DE RESPUESTAS (RAG)
# ============================================================

def responder_con_rag(
    pregunta: str,
    cliente_gemini: genai.Client,
    cliente_groq: Groq,
    coleccion: Any
) -> None:
    print("\nBuscando información relevante...")
    fragmentos = recuperar_fragmentos(
        pregunta,
        cliente_gemini,
        coleccion
    )

    if not fragmentos:
        print("La colección RAG está vacía. Indexa documentos primero.")
        return

    contexto = construir_contexto(fragmentos)

    print(f"Fragmentos recuperados: {len(fragmentos)}")
    print("Generando respuesta con IA...\n")

    instrucciones = (
        INSTRUCCIONES_SISTEMA
        + "\n\n"
        + "Responde utilizando exclusivamente la información del CONTEXTO. "
        + "No inventes datos ni completes información ausente con suposiciones. "
        + "Si el contexto no permite responder, indícalo claramente. "
        + "Distingue los hechos explícitos de las inferencias. "
        + "Cuando sea posible, identifica el documento del que procede cada dato."
    )

    mensaje_usuario = (
        "CONTEXTO DOCUMENTAL:\n"
        f"{contexto}\n\n"
        "PREGUNTA DEL USUARIO:\n"
        f"{pregunta}"
    )

    try:
        respuesta = cliente_groq.chat.completions.create(
            model=MODELO_IA,
            messages=[
                {"role": "system", "content": instrucciones},
                {"role": "user", "content": mensaje_usuario}
            ],
            temperature=0.2
        )

        contenido = respuesta.choices[0].message.content

        print("=" * 60)
        print("RESPUESTA")
        print("=" * 60)
        print(contenido or "El modelo no ha devuelto texto.")

        print("\n" + "=" * 60)
        print("FUENTES RECUPERADAS")
        print("=" * 60)

        fuentes_vistas = set()
        for fragmento in fragmentos:
            clave = (
                fragmento["documento"],
                fragmento["numero_chunk"]
            )
            if clave in fuentes_vistas:
                continue

            fuentes_vistas.add(clave)
            print(
                f"- {fragmento['documento']} "
                f"(fragmento {fragmento['numero_chunk']}, "
                f"distancia {fragmento['distancia']:.4f})"
            )

    except Exception as error:
        print(f"No se pudo generar la respuesta con Groq: {error}")


# ============================================================
# MENÚ PRINCIPAL
# ============================================================

def mostrar_menu() -> None:
    print("\n" + "=" * 60)
    print("SISTEMA RAG - CONSULTA INTELIGENTE DE DOCUMENTOS")
    print("=" * 60)
    print("1. Indexar o actualizar documentos")
    print("2. Preguntar a los documentos (RAG)")
    print("0. Salir")


def main() -> None:
    try:
        cliente_gemini, cliente_groq = crear_clientes()
        coleccion = obtener_coleccion()
    except Exception as error:
        print(f"Error inicializando el sistema: {error}")
        return

    print("Sistema RAG inicializado.")
    print(f"Carpeta de documentos: {RUTA_DOCUMENTOS}")
    print(f"Colección ChromaDB: {NOMBRE_COLECCION}")
    print(f"Fragmentos indexados: {coleccion.count()}")

    while True:
        mostrar_menu()
        opcion = input("\nSelecciona una opción: ").strip()

        if opcion == "1":
            indexar_documentos(cliente_gemini, coleccion)

        elif opcion == "2":
            if coleccion.count() == 0:
                print(
                    "\nNo hay documentos indexados. "
                    "Selecciona primero la opción 1."
                )
                continue

            pregunta = input("\n¿Qué quieres preguntar? ").strip()

            if pregunta:
                responder_con_rag(
                    pregunta,
                    cliente_gemini,
                    cliente_groq,
                    coleccion
                )
            else:
                print("Debes introducir una pregunta.")

        elif opcion == "0":
            print("\nHasta pronto.")
            break

        else:
            print("Opción no válida. Elige 1, 2 o 0.")


if __name__ == "__main__":
    main()
