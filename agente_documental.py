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
    }
]

def ejecutar_herramienta(nombre, argumentos):

    if nombre == "consultar_documentos":

        pregunta = argumentos["pregunta"]

        return consultar_automaticamente(pregunta)

    return f"Herramienta desconocida: {nombre}"

def ejecutar_agente(pregunta, max_iteraciones=5):

    try:
        _, cliente_groq = rag.crear_clientes()

        mensajes = [
            {
                "role": "system",
                "content": (
                "Eres un agente especializado en consultar documentación. "
                "Tienes acceso a la herramienta consultar_documentos. "
                "\n\n"
                "REGLAS DE COMPORTAMIENTO:\n"
                "1. Si el usuario pregunta sobre currículums o documentación "
                "científica, debes utilizar consultar_documentos antes de responder.\n"
                "2. No respondas estas preguntas utilizando tus conocimientos "
                "generales.\n"
                "3. Espera el resultado de la herramienta antes de elaborar "
                "tu respuesta.\n"
                "4. Basa tu respuesta exclusivamente en el resultado obtenido.\n"
                "5. Si la documentación no contiene información suficiente, "
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
    Devuelve los nombres de los documentos indexados.
    """
    try:
        coleccion = rag.obtener_coleccion()

        documentos = rag.obtener_documentos_indexados(coleccion)

        if not documentos:
            return "No hay documentos indexados."

        return "\n".join(documentos)

    except Exception as e:
        return f"Error al listar documentos: {e}"

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
    pregunta = "Busca currículums con experiencia en SQL Server"

    respuesta = ejecutar_agente(pregunta)

    print("\n" + "=" * 60)
    print("RESPUESTA FINAL DEL AGENTE")
    print("=" * 60)
    print(respuesta)