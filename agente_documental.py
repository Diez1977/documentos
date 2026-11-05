import buscador_ia_rag as rag
import chromadb
import json


#Indicamos qué colecciones de Chroma usaremos en el agente
COLECCIONES = {
    "ciencia": "documentos_rag",
    "cv": "buscador_curriculums"
}



HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "consultar_documentos",
            "description": (
                "Consulta los documentos disponibles para responder "
                "preguntas sobre currículums o documentación científica."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pregunta": {
                        "type": "string",
                        "description": "Pregunta que se debe consultar en los documentos."
                    }
                },
                "required": ["pregunta"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resumir_documento",
            "description": (
                "Genera un resumen breve de un documento concreto. "
                "Utiliza esta herramienta cuando el usuario solicite "
                "resumir un documento identificado por su nombre."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre_documento": {
                        "type": "string",
                        "description": "Nombre exacto del documento que se quiere resumir."
                    }
                },
                "required": ["nombre_documento"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "listar_documentos",
            "description": (
                "Lista los documentos que están indexados en las colecciones "
                "de ciencia y currículums. Utiliza esta herramienta cuando "
                "el usuario pregunte qué documentos hay disponibles, "
                "cuáles están indexados o solicite un listado."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    }
]


def ejecutar_herramienta(nombre, argumentos):

    if nombre == "consultar_documentos":

        pregunta = argumentos["pregunta"]

        return consultar_automaticamente(pregunta)

    elif nombre == "listar_documentos":

        return listar_documentos()

    elif nombre == "resumir_documento":

        nombre_documento = argumentos["nombre_documento"]

        return resumir_documento(nombre_documento)

    return f"Herramienta desconocida: {nombre}"


def ejecutar_agente(pregunta, max_iteraciones=5):

    try:
        _, cliente_groq = rag.crear_clientes()

        mensajes = [
            {
                "role": "system",
                "content": (
            "Eres un agente especializado en consultar documentación.\n"
            "Tienes acceso a tres herramientas:\n"
            "- consultar_documentos: consulta el contenido de los documentos.\n"
            "- listar_documentos: muestra los documentos indexados.\n"
            "- resumir_documento: genera un resumen de un documento concreto.\n\n"
        
            "REGLAS DE COMPORTAMIENTO:\n"
            "1. Si el usuario pregunta sobre currículums o documentación "
            "científica, utiliza consultar_documentos.\n"
            "2. Si pregunta qué documentos hay disponibles, utiliza "
            "listar_documentos.\n"
            "3. Si solicita un resumen de un documento concreto, utiliza "
            "resumir_documento.\n"
            "4. No respondas preguntas documentales utilizando tus "
            "conocimientos generales.\n"
            "5. Espera el resultado de la herramienta antes de responder.\n"
            "6. Basa tus respuestas en los resultados obtenidos.\n"
            "7. Si la documentación no contiene información suficiente, "
            "indícalo claramente.\n"
            )
            },
            {
                "role": "user",
                "content": pregunta
            }
        ]

        for _ in range(max_iteraciones):

            respuesta = cliente_groq.chat.completions.create(
                model=rag.MODELO_IA,
                messages=mensajes,
                tools=HERRAMIENTAS,
                tool_choice="auto"
            )

            mensaje = respuesta.choices[0].message

            # Si Groq no solicita herramientas, devuelve su respuesta.
            if not mensaje.tool_calls:
                return mensaje.content

            # Añadir la respuesta de Groq al historial.
            mensajes.append(
                mensaje.model_dump(exclude_none=True)
            )

            # Ejecutar las herramientas solicitadas.
            for llamada in mensaje.tool_calls:

                nombre = llamada.function.name

                argumentos = json.loads(
                    llamada.function.arguments
                )

                resultado = ejecutar_herramienta(
                    nombre,
                    argumentos
                )

                mensajes.append({
                    "role": "tool",
                    "tool_call_id": llamada.id,
                    "content": str(resultado)
                })

        return "Se ha alcanzado el límite de iteraciones del agente."

    except Exception as e:
        return f"Error en el agente: {e}"


def listar_documentos():
    """
    Lista los documentos indexados en todas las colecciones.
    """

    try:
        cliente_chroma = obtener_cliente_chroma()

        respuesta = []

        for tipo, nombre_coleccion in COLECCIONES.items():

            coleccion = cliente_chroma.get_collection(
                name=nombre_coleccion
            )

            datos = coleccion.get(
                include=["metadatas"]
            )

            documentos = set()

            for metadata in datos["metadatas"]:
                nombre = metadata.get("documento_origen")

                if isinstance(nombre, str) and nombre.strip():
                    documentos.add(nombre.strip())

            respuesta.append(f"\nCOLECCIÓN: {tipo.upper()}")
            respuesta.append(f"Documentos: {len(documentos)}")

            for documento in sorted(documentos):
                respuesta.append(f"- {documento}")

        return "\n".join(respuesta)

    except Exception as e:
        return f"Error al listar documentos: {e}"

def resumir_documento(nombre_documento):
    """
    Recupera todos los fragmentos de un documento
    y genera un resumen breve con Groq.
    """

    try:
        cliente_chroma = obtener_cliente_chroma()

        coincidencias = []

        # Buscar el documento en todas las colecciones.
        for tipo, nombre_coleccion in COLECCIONES.items():

            coleccion = cliente_chroma.get_collection(
                name=nombre_coleccion
            )

            datos = coleccion.get(
                where={"documento_origen": nombre_documento},
                include=["documents", "metadatas"]
            )

            if datos["documents"]:
                coincidencias.append({
                    "tipo": tipo,
                    "documentos": datos["documents"],
                    "metadatos": datos["metadatas"]
                })

        if not coincidencias:
            return (
                f"No se ha encontrado el documento "
                f"'{nombre_documento}'."
            )

        if len(coincidencias) > 1:
            tipos = ", ".join(
                coincidencia["tipo"]
                for coincidencia in coincidencias
            )
            return (
                f"El documento aparece en varias colecciones: {tipos}. "
                "Necesitamos identificar cuál se quiere resumir."
            )

        coincidencia = coincidencias[0]

        # Ordenar los fragmentos por su número original.
        fragmentos = sorted(
            zip(
                coincidencia["metadatos"],
                coincidencia["documentos"]
            ),
            key=lambda elemento: int(
                elemento[0].get("numero_chunk", 0)
            )
        )

        textos = [
            texto for _, texto in fragmentos
            if texto and texto.strip()
        ]

        if not textos:
            return "El documento no contiene texto para resumir."

        print(
            f"Documento: {nombre_documento} | "
            f"Fragmentos: {len(textos)}"
        )

        _, cliente_groq = rag.crear_clientes()

        # Dividir el contenido en bloques para controlar
        # el tamaño de cada petición a Groq.
        bloques = []
        bloque_actual = ""

        for texto in textos:

            if bloque_actual and len(bloque_actual) + len(texto) > 12000:
                bloques.append(bloque_actual)
                bloque_actual = ""

            bloque_actual += texto + "\n\n"

        if bloque_actual:
            bloques.append(bloque_actual)

        resumenes = []

        # Resumir cada bloque.
        for numero, bloque in enumerate(bloques, start=1):

            respuesta = cliente_groq.chat.completions.create(
                model=rag.MODELO_IA,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Resume el contenido documental de forma "
                            "breve y objetiva. Utiliza exclusivamente "
                            "la información proporcionada. "
                            "No inventes ni añadas datos."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Documento: {nombre_documento}\n"
                            f"Parte {numero} de {len(bloques)}:\n\n"
                            f"{bloque}"
                        )
                    }
                ],
                temperature=0.2,
                max_tokens=500
            )

            resumen = respuesta.choices[0].message.content

            if not resumen:
                return "Groq no ha generado el resumen."

            resumenes.append(resumen)

        # Si hay varios bloques, unificar sus resúmenes.
        if len(resumenes) == 1:
            return resumenes[0]

        respuesta_final = cliente_groq.chat.completions.create(
            model=rag.MODELO_IA,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Elabora un resumen breve y unificado "
                        "a partir de los resúmenes parciales. "
                        "Evita repeticiones y no añadas información."
                    )
                },
                {
                    "role": "user",
                    "content": "\n\n".join(resumenes)
                }
            ],
            temperature=0.2,
            max_tokens=700
        )

        return (
            respuesta_final.choices[0].message.content
            or "No se ha podido generar el resumen final."
        )

    except Exception as e:
        return f"Error al resumir el documento: {e}"

def buscar_documentos(pregunta, tipo_documento):
    """
    Busca documentos en la colección indicada.
    """
    try:
        if tipo_documento not in COLECCIONES:
            return "Tipo de documento no válido."

        cliente_chroma = obtener_cliente_chroma()

        coleccion = cliente_chroma.get_collection(
            name=COLECCIONES[tipo_documento]
        )

        cliente_gemini, _ = rag.crear_clientes()

        fragmentos = rag.recuperar_fragmentos(
            pregunta,
            cliente_gemini,
            coleccion
        )

        if not fragmentos:
            return "No se han encontrado documentos relevantes."

        documentos = {}

        for fragmento in fragmentos:
            nombre = fragmento["documento"]
            distancia = fragmento["distancia"]

            if nombre not in documentos or distancia < documentos[nombre]:
                documentos[nombre] = distancia

        resultados = sorted(
            documentos.items(),
            key=lambda x: x[1]
        )

        respuesta = ["Documentos encontrados:"]

        for nombre, distancia in resultados:
            respuesta.append(
                f"- {nombre} (distancia: {distancia:.4f})"
            )

        return "\n".join(respuesta)

    except Exception as e:
        return f"Error en la búsqueda: {e}"

# Obtenemos cliente Chroma
def obtener_cliente_chroma():
    return chromadb.PersistentClient(
        path=str(rag.RUTA_CHROMA)
    )
# Comprobar las  oolecciones existentes en Chroma
def comprobar_colecciones():
    cliente = obtener_cliente_chroma()

    nombres = [
        "documentos_python_chunks",
        "buscador_curriculums",
        "documentos_rag",
        "documentos_python",
        "buscador_documentos"
    ]

    for nombre in nombres:
        coleccion = cliente.get_collection(name=nombre)

        print("\n" + "=" * 50)
        print(f"COLECCIÓN: {nombre}")
        print(f"FRAGMENTOS: {coleccion.count()}")

        datos = coleccion.get(
            limit=3,
            include=["metadatas", "documents"]
        )

        for metadata, texto in zip(
            datos["metadatas"],
            datos["documents"]
        ):
            print(f"METADATOS: {metadata}")
            print(f"TEXTO: {texto[:150]}")

def seleccionar_coleccion(pregunta):
    """
    Decide qué colección utilizar según la pregunta.
    """
    try:
        _, cliente_groq = rag.crear_clientes()

        respuesta = cliente_groq.chat.completions.create(
            model=rag.MODELO_IA,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Clasifica la petición del usuario en una categoría.

                    - cv: currículums, candidatos, experiencia laboral,
                      formación profesional, perfiles profesionales.
                    - ciencia: física, mecánica cuántica y documentos
                      científicos.

                    Responde exclusivamente con: cv o ciencia.
                    Si no corresponde claramente a ninguna, responde: ninguna.
                    """
                },
                {
                    "role": "user",
                    "content": pregunta
                }
            ],
            temperature=0,
            max_tokens=150
        )

        resultado = respuesta.choices[0].message.content.strip().lower()



        if resultado in COLECCIONES:
            return resultado

        return None

    except Exception as e:
        print(f"Error seleccionando colección: {e}")
        return None

def buscar_automaticamente(pregunta):
    tipo_documento = seleccionar_coleccion(pregunta)

    if tipo_documento is None:
        return "No puedo determinar qué colección utilizar."

    print(f"Colección seleccionada: {tipo_documento}")

    return buscar_documentos(pregunta, tipo_documento)

def preguntar_documentos(pregunta, tipo_documento):

    try:
        if tipo_documento not in COLECCIONES:
            return "Tipo de documento no válido."

        # Obtener la colección
        cliente_chroma = obtener_cliente_chroma()

        coleccion = cliente_chroma.get_collection(
            name=COLECCIONES[tipo_documento]
        )

        # Crear clientes
        cliente_gemini, cliente_groq = rag.crear_clientes()

        # Recuperar fragmentos
        fragmentos = rag.recuperar_fragmentos(
            pregunta,
            cliente_gemini,
            coleccion
        )

        if not fragmentos:
            return "No se ha encontrado información suficientemente relevante."

        # Construir contexto
        contexto = rag.construir_contexto(fragmentos)

        # Construir mensaje
        mensaje_usuario = (
            "A continuación tienes los fragmentos recuperados "
            "de los documentos.\n\n"
            "=== INICIO DEL CONTEXTO DOCUMENTAL ===\n"
            f"{contexto}\n"
            "=== FIN DEL CONTEXTO DOCUMENTAL ===\n\n"
            "PREGUNTA DEL USUARIO:\n"
            f"{pregunta}"
        )

        # Consultar a Groq
        respuesta = cliente_groq.chat.completions.create(
            model=rag.MODELO_IA,
            messages=[
                {
                    "role": "system",
                    "content": rag.INSTRUCCIONES_SISTEMA
                },
                {
                    "role": "user",
                    "content": mensaje_usuario
                }
            ],
            temperature=0.2
        )

        contenido = respuesta.choices[0].message.content

        # Validar referencias
        rag.validar_referencias(contenido, fragmentos)

        return contenido

    except Exception as e:
        return f"Error al consultar los documentos: {e}"

def consultar_automaticamente(pregunta):
    """
    Clasifica la pregunta, selecciona la colección
    y genera una respuesta basada en sus documentos.
    """

    tipo_documento = seleccionar_coleccion(pregunta)

    if tipo_documento is None:
        return "No puedo determinar qué colección utilizar."

    print(f"Colección seleccionada: {tipo_documento}")

    respuesta = preguntar_documentos(
        pregunta,
        tipo_documento
    )

    return respuesta

if __name__ == "__main__":
    pregunta = input("\nTú: ")

    respuesta = ejecutar_agente(pregunta)

    print("\n" + "=" * 60)
    print("RESPUESTA FINAL DEL AGENTE")
    print("=" * 60)
    print(respuesta)
    
