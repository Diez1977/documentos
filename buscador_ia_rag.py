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
import re
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
RESULTADOS_RECUPERACION = 20
FRAGMENTOS_CONTEXTO = 5
MAX_FRAGMENTOS_POR_DOCUMENTO = 2
DISTANCIA_MAXIMA = 0.38
MAX_CARACTERES_CONTEXTO = 18000
PAUSA_ENTRE_EMBEDDINGS = 1.0
MAX_REINTENTOS = 3
MAX_CARACTERES_BLOQUE = 12000
MAX_CARACTERES_RESUMENES = 16000

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

def agrupar_fragmentos(
    fragmentos: list[dict[str, Any]],
    max_caracteres: int = MAX_CARACTERES_BLOQUE
) -> list[list[dict[str, Any]]]:

    bloques = []
    bloque_actual = []
    caracteres = 0

    for fragmento in fragmentos:

        cabecera = (
            f"[Fuente: {fragmento['documento']} | "
            f"Fragmento: {fragmento['numero_chunk']}]\n"
        )

        longitud = len(cabecera) + len(fragmento["texto"]) + 2

        # Si un fragmento supera el límite, lo dividimos
        # conservando su número de referencia.
        if longitud > max_caracteres:

            if bloque_actual:
                bloques.append(bloque_actual)
                bloque_actual = []
                caracteres = 0

            texto = fragmento["texto"]
            tamano = max_caracteres - len(cabecera) - 2

            for inicio in range(0, len(texto), tamano):

                parte = texto[inicio:inicio + tamano]

                bloques.append([{
                    **fragmento,
                    "texto": parte
                }])

            continue

        if caracteres + longitud > max_caracteres:

            bloques.append(bloque_actual)
            bloque_actual = []
            caracteres = 0

        bloque_actual.append(fragmento)
        caracteres += longitud

    if bloque_actual:
        bloques.append(bloque_actual)

    return bloques
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

    ids = []
    documentos = []
    metadatos = []
    embeddings = []

    # 1. Generamos todos los embeddings antes de modificar ChromaDB.
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

    # 2. Recuperamos los identificadores anteriores.
    resultado_anterior = coleccion.get(
        where={"documento_origen": nombre}
    )

    ids_antiguos = resultado_anterior["ids"]

    # 3. Insertamos o actualizamos los fragmentos nuevos.
    coleccion.upsert(
        ids=ids,
        documents=documentos,
        metadatas=metadatos,
        embeddings=embeddings
    )

    # 4. Eliminamos los fragmentos antiguos que sobren.
    ids_obsoletos = [
        id_antiguo
        for id_antiguo in ids_antiguos
        if id_antiguo not in ids
    ]

    if ids_obsoletos:
        coleccion.delete(ids=ids_obsoletos)

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
    contador_documentos = {}

    for texto, metadata, distancia in zip(
        documentos,
        metadatos,
        distancias
    ):
        distancia = float(distancia)
        print(
            f"Documento: {metadata.get('documento_origen')} | "
            f"Fragmento: {metadata.get('numero_chunk')} | "
            f"Distancia: {distancia:.4f}"
        )

        if distancia > DISTANCIA_MAXIMA:
            continue

        nombre_documento = metadata.get(
            "documento_origen",
            "Desconocido"
        )

        cantidad_documento = contador_documentos.get(
            nombre_documento,
            0
        )

        if cantidad_documento >= MAX_FRAGMENTOS_POR_DOCUMENTO:
            continue

        fragmentos.append({
            "texto": texto,
            "documento": nombre_documento,
            "numero_chunk": metadata.get("numero_chunk", "?"),
            "distancia": distancia
        })

        contador_documentos[nombre_documento] = cantidad_documento + 1

    return fragmentos



def construir_contexto(
    fragmentos: list[dict[str, Any]],
    max_caracteres: int = MAX_CARACTERES_CONTEXTO
) -> str:
    partes = []
    caracteres = 0

    for fragmento in fragmentos[:FRAGMENTOS_CONTEXTO]:
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

def construir_contexto_documento(
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

def obtener_documentos_indexados(coleccion: Any) -> list[str]:

    resultado = coleccion.get(include=["metadatas"])

    documentos = {
        metadata["documento_origen"]
        for metadata in resultado["metadatas"]
        if metadata and "documento_origen" in metadata
    }

    return sorted(documentos)

def seleccionar_documento(coleccion: Any) -> str | None:

    documentos = obtener_documentos_indexados(coleccion)

    if not documentos:
        print("No hay documentos indexados.")
        return None

    print("\nDOCUMENTOS DISPONIBLES")
    print("=" * 60)

    for numero, documento in enumerate(documentos, start=1):
        print(f"{numero}. {documento}")

    try:
        opcion = int(input("\nSelecciona un documento: "))

        if 1 <= opcion <= len(documentos):
            return documentos[opcion - 1]

        print("Opción no válida.")
        return None

    except ValueError:
        print("Debes introducir un número.")
        return None


def seleccionar_dos_documentos(
    coleccion: Any
) -> tuple[str, str] | None:

    documentos = obtener_documentos_indexados(coleccion)

    if len(documentos) < 2:
        print("Se necesitan al menos dos documentos indexados.")
        return None

    print("\nDOCUMENTOS DISPONIBLES")
    print("=" * 60)

    for numero, documento in enumerate(documentos, start=1):
        print(f"{numero}. {documento}")

    try:
        opcion1 = int(input("\nSelecciona el primer documento: "))

        if not 1 <= opcion1 <= len(documentos):
            print("Primera opción no válida.")
            return None

        opcion2 = int(input("Selecciona el segundo documento: "))

        if not 1 <= opcion2 <= len(documentos):
            print("Segunda opción no válida.")
            return None

        if opcion1 == opcion2:
            print("Debes seleccionar dos documentos diferentes.")
            return None

        return documentos[opcion1 - 1], documentos[opcion2 - 1]

    except ValueError:
        print("Debes introducir números.")
        return None



def comparar_documentos(
    cliente_groq: Groq,
    coleccion: Any
) -> None:

    seleccion = seleccionar_dos_documentos(coleccion)

    if seleccion is None:
        return

    documento1, documento2 = seleccion

    fragmentos1 = recuperar_documento_completo(
        coleccion,
        documento1
    )

    fragmentos2 = recuperar_documento_completo(
        coleccion,
        documento2
    )

    if not fragmentos1 or not fragmentos2:
        print("No se ha podido recuperar el contenido de ambos documentos.")
        return

    try:

        print("\n" + "=" * 60)
        print("COMPARACIÓN DE DOCUMENTOS")
        print("=" * 60)

        print(f"\nDocumento 1: {documento1}")
        print(f"Documento 2: {documento2}")

        print("\nGenerando síntesis del primer documento...")
        sintesis1 = generar_sintesis_documento(
            cliente_groq,
            fragmentos1
        )

        print("\nGenerando síntesis del segundo documento...")
        sintesis2 = generar_sintesis_documento(
            cliente_groq,
            fragmentos2
        )

        print("\nComparando documentos...\n")

        instrucciones = (
            "Eres un asistente especializado en comparar documentos. "
            "Analiza exclusivamente las dos síntesis proporcionadas. "
            "No inventes información ni extraigas conclusiones que "
            "no estén respaldadas por ellas. "
            "Organiza el resultado en estas cuatro secciones:\n"
            "1. COINCIDENCIAS: temas, conceptos o información común.\n"
            "2. DIFERENCIAS: contenidos o enfoques diferentes.\n"
            "3. INFORMACIÓN EXCLUSIVA: aspectos relevantes que aparecen "
            "en un documento y no en el otro. Identifica cuál.\n"
            "4. CONCLUSIÓN: valoración breve de la relación entre ambos.\n\n"
            "Incluye referencias documentales exactas para respaldar "
            "las afirmaciones, utilizando exclusivamente este formato:\n"
            "[Fuente: nombre_documento | Fragmento: numero].\n"
            "No inventes ni modifiques referencias. "
            "Si no hay información suficiente para establecer una "
            "coincidencia o diferencia, indícalo."
        )

        contenido = (
            f"DOCUMENTO 1: {documento1}\n\n"
            f"{sintesis1}\n\n"
            f"DOCUMENTO 2: {documento2}\n\n"
            f"{sintesis2}"
        )

        resultado = generar_resumen_groq(
            cliente_groq,
            instrucciones,
            contenido
        )

        print("\n" + "=" * 60)
        print("RESULTADO DE LA COMPARACIÓN")
        print("=" * 60)
        print(resultado)

        validar_referencias(
            resultado,
            fragmentos1 + fragmentos2
        )

    except Exception as e:
        print(f"Error al comparar los documentos: {e}")

def generar_sintesis_documento(
    cliente_groq: Groq,
    fragmentos: list[dict[str, Any]]
) -> str:

    bloques = agrupar_fragmentos(fragmentos)
    resumenes = []

    instrucciones = (
        "Resume exclusivamente la información proporcionada. "
        "Identifica las ideas principales y los conceptos importantes. "
        "No inventes información. "
        "Sé conciso y utiliza como máximo 600 palabras. "
        "Prioriza los datos más relevantes y evita explicaciones secundarias. "
        "Conserva las referencias documentales exactas con este formato: "
        "[Fuente: nombre_documento | Fragmento: numero]. "
        "No inventes ni modifiques referencias."
    )
    # Generar una síntesis por bloque.
    for numero, bloque in enumerate(bloques, start=1):

        print(f"  Procesando bloque {numero} de {len(bloques)}...")

        contexto = construir_contexto_documento(
            bloque,
            max_caracteres=MAX_CARACTERES_BLOQUE
        )

        resumen = generar_resumen_groq(
            cliente_groq,
            instrucciones,
            contexto
        )

        resumenes.append(resumen)

    # Reducir las síntesis si superan el límite establecido.
    limite = MAX_CARACTERES_RESUMENES // 2

    while sum(len(r) for r in resumenes) > limite:

        grupos = []
        grupo_actual = []
        caracteres = 0

        for resumen in resumenes:

            longitud = len(resumen) + 2

            if grupo_actual and caracteres + longitud > limite:
                grupos.append(grupo_actual)
                grupo_actual = []
                caracteres = 0

            grupo_actual.append(resumen)
            caracteres += longitud

        if grupo_actual:
            grupos.append(grupo_actual)

        nuevas_sintesis = []

        for grupo in grupos:

            contenido = "\n\n".join(grupo)

            sintesis = generar_resumen_groq(
                cliente_groq,
                (
                    "Consolida los resúmenes proporcionados. "
                    "Elimina repeticiones y conserva la información relevante. "
                    "Mantén exactamente todas las referencias documentales "
                    "que respalden la información. "
                    "No inventes datos ni referencias."
                ),
                contenido
            )

            nuevas_sintesis.append(sintesis)

        resumenes = nuevas_sintesis

    return "\n\n".join(resumenes)


def recuperar_documento_completo(
    coleccion: Any,
    nombre_documento: str
) -> list[dict[str, Any]]:

    resultado = coleccion.get(
        where={"documento_origen": nombre_documento},
        include=["documents", "metadatas"]
    )

    fragmentos = []

    for texto, metadata in zip(
        resultado["documents"],
        resultado["metadatas"]
    ):
        fragmentos.append({
            "texto": texto,
            "documento": nombre_documento,
            "numero_chunk": metadata["numero_chunk"]
        })

    fragmentos.sort(
        key=lambda fragmento: fragmento["numero_chunk"]
    )

    return fragmentos
def listar_documentos_con_resumen(
    cliente_groq: Groq,
    coleccion: Any
) -> None:

    documentos = obtener_documentos_indexados(coleccion)

    if not documentos:
        print("No hay documentos indexados.")
        return

    print("\nDOCUMENTOS Y DESCRIPCIONES")
    print("=" * 60)

    for numero, nombre in enumerate(documentos, start=1):

        fragmentos = recuperar_documento_completo(
            coleccion,
            nombre
        )

        contexto = construir_contexto_documento(fragmentos)

        try:
            respuesta = cliente_groq.chat.completions.create(
                model=MODELO_IA,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Describe el contenido del documento "
                            "en un máximo de dos frases. "
                            "Utiliza exclusivamente el contexto "
                            "proporcionado. No inventes información."
                        )
                    },
                    {
                        "role": "user",
                        "content": contexto
                    }
                ],
                temperature=0.2
            )

            resumen = respuesta.choices[0].message.content

            print(f"\n{numero}. {nombre}")
            print(f"   {resumen or 'No se pudo generar la descripción.'}")

        except Exception as e:
            print(f"\n{numero}. {nombre}")
            print(f"   Error al generar la descripción: {e}")

    print("\n" + "=" * 60)

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

    # 1. Recuperar fragmentos relevantes
    fragmentos = recuperar_fragmentos(
        pregunta,
        cliente_gemini,
        coleccion
    )

    # 2. Comprobar si hay información disponible
    if not fragmentos:

        if coleccion.count() == 0:
            print("La colección RAG está vacía. Indexa documentos primero.")

        else:
            print("No se ha encontrado información suficientemente relevante.")
            print(f"Distancia máxima permitida: {DISTANCIA_MAXIMA}")
            print("No se realizará ninguna consulta a Groq.")

        return

    # 3. Construir el contexto documental
    contexto = construir_contexto(fragmentos)

    print("Generando respuesta con IA...\n")

    # 4. Instrucciones del sistema
    instrucciones = INSTRUCCIONES_SISTEMA

    # 5. Construir el mensaje del usuario
    mensaje_usuario = (
        "A continuación tienes los fragmentos recuperados "
        "de los documentos.\n\n"
        "=== INICIO DEL CONTEXTO DOCUMENTAL ===\n"
        f"{contexto}\n"
        "=== FIN DEL CONTEXTO DOCUMENTAL ===\n\n"
        "PREGUNTA DEL USUARIO:\n"
        f"{pregunta}"
    )

    # 6. Consultar a Groq
    try:

        respuesta = cliente_groq.chat.completions.create(
            model=MODELO_IA,
            messages=[
                {
                    "role": "system",
                    "content": instrucciones
                },
                {
                    "role": "user",
                    "content": mensaje_usuario
                }
            ],
            temperature=0.2
        )

        # 7. Mostrar respuesta
        contenido = respuesta.choices[0].message.content

        print("=" * 60)
        print("RESPUESTA")
        print("=" * 60)

        print(contenido)
        # Validar referencias
        validar_referencias(contenido, fragmentos)
        # 8. Mostrar fuentes recuperadas
        print("\n" + "=" * 60)
        print("FUENTES RECUPERADAS")
        print("=" * 60)

        for numero, fragmento in enumerate(fragmentos, start=1):

            print(f"\nFuente {numero}")
            print(f"Documento: {fragmento['documento']}")
            print(f"Fragmento: {fragmento['numero_chunk']}")
            print(f"Distancia: {fragmento['distancia']:.4f}")

    except Exception as e:

        print(f"Error al generar la respuesta con IA: {e}")




def validar_referencias(
    respuesta: str,
    fragmentos: list[dict[str, Any]]
) -> None:

    referencias_validas = {
        (f["documento"], str(f["numero_chunk"]))
        for f in fragmentos
    }

    # Documentos disponibles para cada número de fragmento.
    documentos_por_numero = {}

    for documento, numero in referencias_validas:
        documentos_por_numero.setdefault(numero, set()).add(documento)

    # Admite referencias con Fuente opcional y distintos tipos de guion.
    patron = (
        r"\[(?:Fuente:\s*(.*?)\s*\|\s*)?"
        r"Fragmentos?:\s*"
        r"(\d+(?:\s*[-‐‑‒–—]\s*\d+)?"
        r"(?:\s*,\s*\d+(?:\s*[-‐‑‒–—]\s*\d+)?)*)\s*\]"
    )

    coincidencias = re.findall(patron, respuesta)

    referencias_citadas = set()
    referencias_ambiguas = set()

    for documento, numeros in coincidencias:

        # Separar números individuales y rangos.
        partes = re.split(r"\s*,\s*", numeros)

        numeros_extraidos = set()

        for parte in partes:

            rango = re.fullmatch(
                r"(\d+)\s*[-‐‑‒–—]\s*(\d+)",
                parte
            )

            if rango:
                inicio, fin = map(int, rango.groups())

                if inicio > fin:
                    inicio, fin = fin, inicio

                numeros_extraidos.update(
                    str(n) for n in range(inicio, fin + 1)
                )

            else:
                numeros_extraidos.add(parte.strip())

        for numero in numeros_extraidos:

            if documento:
                referencias_citadas.add(
                    (documento.strip(), numero)
                )

            else:
                # Sin documento, solo podemos validar si el número
                # pertenece a un único documento.
                documentos = documentos_por_numero.get(numero, set())

                if len(documentos) == 1:
                    referencias_citadas.add(
                        (next(iter(documentos)), numero)
                    )

                elif len(documentos) > 1:
                    referencias_ambiguas.add(numero)

                else:
                    referencias_citadas.add(
                        ("Documento no identificado", numero)
                    )

    referencias_correctas = referencias_citadas & referencias_validas
    referencias_incorrectas = referencias_citadas - referencias_validas

    print("\n" + "=" * 60)
    print("VALIDACIÓN DE REFERENCIAS")
    print("=" * 60)

    # Mostrar resultados separados por documento.
    documentos = sorted({doc for doc, _ in referencias_validas})

    for documento in documentos:

        correctas = sorted(
            numero for doc, numero in referencias_correctas
            if doc == documento
        )

        incorrectas = sorted(
            numero for doc, numero in referencias_incorrectas
            if doc == documento
        )

        print(f"\nDOCUMENTO: {documento}")
        print("-" * 60)
        print(f"Referencias válidas: {len(correctas)}")
        print(f"Referencias no válidas: {len(incorrectas)}")

        if correctas:
            print("Válidas: " + ", ".join(correctas))

        if incorrectas:
            print("No válidas: " + ", ".join(incorrectas))

    if referencias_ambiguas:
        print("\nREFERENCIAS AMBIGUAS")
        print("-" * 60)

        for numero in sorted(referencias_ambiguas, key=int):
            print(
                f"Fragmento {numero}: aparece en varios documentos "
                "y no se puede identificar su origen."
            )

    if not coincidencias:
        print("\nAVISO: No se han encontrado referencias reconocibles.")

    print("\n" + "-" * 60)
    print(f"Total de referencias válidas: {len(referencias_correctas)}")
    print(f"Total de referencias no válidas: {len(referencias_incorrectas)}")
    print(f"Total de referencias ambiguas: {len(referencias_ambiguas)}")

    if referencias_incorrectas or referencias_ambiguas:
        print("\nAVISO: Hay referencias que requieren revisión.")
    elif referencias_correctas:
        print("\nTodas las referencias identificadas corresponden a fragmentos recuperados.")
    else:
        print("\nNo se han podido validar referencias.")


def resumir_documento(
    cliente_groq: Groq,
    coleccion: Any
) -> None:

    nombre = seleccionar_documento(coleccion)

    if nombre is None:
        return

    fragmentos = recuperar_documento_completo(
        coleccion,
        nombre
    )

    if not fragmentos:
        print("No se han encontrado fragmentos del documento.")
        return

    # FASE 1: Dividir el documento en bloques.
    bloques = agrupar_fragmentos(fragmentos)

    print("\n" + "=" * 60)
    print(f"RESUMEN DEL DOCUMENTO: {nombre}")
    print("=" * 60)
    print(f"Fragmentos recuperados: {len(fragmentos)}")
    print(f"Bloques de procesamiento: {len(bloques)}")
    print("\nGenerando resúmenes parciales...\n")

    try:

        resumenes = []

        # FASE 2: Generar un resumen por cada bloque.
        for numero, bloque in enumerate(bloques, start=1):

            print(f"Procesando bloque {numero} de {len(bloques)}...")

            contexto = construir_contexto_documento(
                bloque,
                max_caracteres=MAX_CARACTERES_BLOQUE
            )

            instrucciones = (
                "Eres un asistente especializado en resumir documentos. "
                "Resume exclusivamente la información proporcionada. "
                "No inventes información. "
                "Identifica las ideas principales y los conceptos importantes. "
                "Sé conciso, pero conserva los datos relevantes. "
                "Cada afirmación importante debe incluir su referencia "
                "documental utilizando EXCLUSIVAMENTE este formato: "
                "[Fuente: nombre_documento | Fragmento: numero]. "
                "Utiliza exactamente los nombres y números presentes "
                "en el contexto. No inventes referencias. "
                "Limita el resumen a unas 1200 palabras como máximo."
            )

            resumen_parcial = generar_resumen_groq(
                cliente_groq,
                instrucciones,
                contexto
            )

            resumenes.append(resumen_parcial)

        # FASE 3: Consolidar los resúmenes parciales si es necesario.
        while sum(len(r) for r in resumenes) > MAX_CARACTERES_RESUMENES:

            print("\nConsolidando resúmenes parciales...")

            grupos = []
            grupo_actual = []
            caracteres = 0

            for resumen in resumenes:

                longitud = len(resumen) + 2

                if (
                    grupo_actual
                    and caracteres + longitud > MAX_CARACTERES_RESUMENES
                ):
                    grupos.append(grupo_actual)
                    grupo_actual = []
                    caracteres = 0

                grupo_actual.append(resumen)
                caracteres += longitud

            if grupo_actual:
                grupos.append(grupo_actual)

            nuevos_resumenes = []

            for grupo in grupos:

                contenido_grupo = "\n\n".join(grupo)

                instrucciones = (
                    "Eres un asistente especializado en consolidar resúmenes. "
                    "Integra la información proporcionada, elimina repeticiones "
                    "y conserva las ideas importantes. "
                    "OBLIGATORIO: conserva exactamente las referencias "
                    "documentales existentes, sin modificarlas ni inventar otras. "
                    "No elimines referencias que respalden información relevante. "
                    "No añadas información que no aparezca en los textos."
                )

                resumen_consolidado = generar_resumen_groq(
                    cliente_groq,
                    instrucciones,
                    contenido_grupo
                )

                nuevos_resumenes.append(resumen_consolidado)

            resumenes = nuevos_resumenes

        # FASE 4: Generar el resumen definitivo.
        print("\nGenerando resumen definitivo...\n")

        contenido_parcial = "\n\n".join(resumenes)

        instrucciones_finales = (
            "Eres un asistente especializado en elaborar resúmenes "
            "de documentos. "
            "Elabora un resumen final claro, estructurado y conciso "
            "a partir de los resúmenes proporcionados. "
            "Identifica las ideas principales, los conceptos importantes "
            "y las conclusiones del documento. "
            "Evita repeticiones. "
            "No inventes información. "
            "OBLIGATORIO: conserva exactamente las referencias documentales "
            "existentes utilizando el formato "
            "[Fuente: nombre_documento | Fragmento: numero]. "
            "No inventes, modifiques ni atribuyas referencias incorrectas."
        )

        contenido = generar_resumen_groq(
            cliente_groq,
            instrucciones_finales,
            contenido_parcial
        )

        print("\n" + "=" * 60)
        print(f"RESUMEN: {nombre}")
        print("=" * 60)
        print(contenido)

        # FASE 5: Validar las referencias contra los fragmentos originales.
        validar_referencias(contenido, fragmentos)

    except Exception as e:
        print(f"Error al generar el resumen: {e}")


def breve_resumen_documento(
    cliente_groq: Groq,
    coleccion: Any
) -> None:

    nombre = seleccionar_documento(coleccion)

    if nombre is None:
        return

    fragmentos = recuperar_documento_completo(
        coleccion,
        nombre
    )

    contexto = construir_contexto_documento(fragmentos)

    try:
        respuesta = cliente_groq.chat.completions.create(
            model=MODELO_IA,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Resume el documento en un máximo de tres frases. "
                        "Identifica su tema principal y su contenido más relevante. "
                        "Utiliza exclusivamente la información proporcionada. "
                        "No inventes información."
                    )
                },
                {
                    "role": "user",
                    "content": contexto
                }
            ],
            temperature=0.2
        )

        print("\n" + "=" * 60)
        print(f"BREVE RESUMEN: {nombre}")
        print("=" * 60)
        print(respuesta.choices[0].message.content)

    except Exception as e:
        print(f"Error al generar el resumen: {e}")

def generar_resumen_groq(
    cliente_groq: Groq,
    instrucciones: str,
    contenido: str
) -> str:

    respuesta = cliente_groq.chat.completions.create(
        model=MODELO_IA,
        messages=[
            {"role": "system", "content": instrucciones},
            {"role": "user", "content": contenido}
        ],
        temperature=0.2
    )

    if not respuesta.choices:
        raise ValueError("Groq no ha devuelto ninguna opción.")

    opcion = respuesta.choices[0]
    mensaje = opcion.message
    resultado = mensaje.content

    if not resultado:
        print("\nDIAGNÓSTICO DE GROQ")
        print("-" * 40)
        print(f"Modelo: {respuesta.model}")
        print(f"Motivo de finalización: {opcion.finish_reason}")
        print(f"Contenido: {resultado!r}")
        print(f"Rechazo: {mensaje.refusal}")
        print(f"Longitud del razonamiento: {len(mensaje.reasoning or '') if hasattr(mensaje, 'reasoning') else 'No disponible'}")
        raise ValueError("Groq ha devuelto una respuesta vacía.")

    return resultado

# ============================================================
# MENÚ PRINCIPAL
# ============================================================

def mostrar_menu() -> None:
    print("\n" + "=" * 60)
    print("SISTEMA RAG - CONSULTA INTELIGENTE DE DOCUMENTOS")
    print("=" * 60)
    print("1. Indexar o actualizar documentos")
    print("2. Preguntar a los documentos (RAG)")
    print("3. Resumir un documento")
    print("4. Listar documentos con breve resumen")
    print("5. Breve resumen de un documento")
    print("6. Comparar dos documentos")
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

        elif opcion == "3":
            if coleccion.count() == 0:
                print("\nNo hay documentos indexados.")
                continue

            resumir_documento(cliente_groq, coleccion)
        elif opcion == "4":
            listar_documentos_con_resumen(
                cliente_groq,
                coleccion
            )
        elif opcion == "5":
            breve_resumen_documento(cliente_groq, coleccion)
        elif opcion == "6":
            if coleccion.count() < 2:
                print("\nSe necesitan al menos dos documentos indexados.")
                continue

            comparar_documentos(cliente_groq, coleccion)

        elif opcion == "0":
            print("\nHasta pronto.")
            break

        else:
            print("Opción no válida. Elige 1, 2 o 0.")


if __name__ == "__main__":
    main()
