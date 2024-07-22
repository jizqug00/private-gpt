"""This file should be imported if and only if you want to run the UI locally."""

import itertools
import logging
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

import json
import os
import uuid
import tkinter as tk
from tkinter import *
import datetime
import requests
from bs4 import BeautifulSoup
from tkinter import messagebox
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
user_id = "a"
all_messages_g = ""
user_name = ""
historial_str = ""

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "My Private GPT"

SOURCES_SEPARATOR = "\n\n Sources: \n"

MODES = ["Query Files", "Search Files", "LLM Chat (no context from files)"]


class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> list["Source"]:
        curated_sources = []

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.append(source)
            curated_sources = list(
                dict.fromkeys(curated_sources).keys()
            )  # Unique sources only

        return curated_sources


@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service

        # Cache the UI blocks
        self._ui_block = None

        self._selected_filename = None

        # Initialize system prompt based on default mode
        self.mode = MODES[0]
        self._system_prompt = self._get_default_system_prompt(self.mode)

    def _chat(self, message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response
                time.sleep(0.02)

            if completion_gen.sources:
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
                sources_text = "\n\n\n"
                used_files = set()
                for index, source in enumerate(cur_sources, start=1):
                    if f"{source.file}-{source.page}" not in used_files:
                        sources_text = (
                            sources_text
                            + f"{index}. {source.file} (page {source.page}) \n\n"
                        )
                        used_files.add(f"{source.file}-{source.page}")
                full_response += sources_text
            yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = list(
                itertools.chain(
                    *[
                        [
                            ChatMessage(content=interaction[0], role=MessageRole.USER),
                            ChatMessage(
                                # Remove from history content the Sources information
                                content=interaction[1].split(SOURCES_SEPARATOR)[0],
                                role=MessageRole.ASSISTANT,
                            ),
                        ]
                        for interaction in history
                    ]
                )
            )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)

        global all_messages_g
        all_messages_g = build_history()

        if(message=="/Guardar" and user_id!="a"):
            self.actualizar_historial()

        all_messages = [*build_history(), new_message]
        # If a system prompt is set, add it as a system message
        if self._system_prompt:
            all_messages.insert(
                0,
                ChatMessage(
                    content=self._system_prompt,
                    role=MessageRole.SYSTEM,
                ),
            )
        match mode:
            case "Query Files":

                # Use only the selected file for the query
                context_filter = None
                if self._selected_filename is not None:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                )
                yield from yield_deltas(query_stream)
            case "LLM Chat (no context from files)":
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                )
                yield from yield_deltas(llm_stream)

            case "Search Files":
                response = self._chunks_service.retrieve_relevant(
                    text=message, limit=4, prev_next_chunks=0
                )

                sources = Source.curate_sources(response)

                yield "\n\n\n".join(
                    f"{index}. **{source.file} "
                    f"(page {source.page})**\n "
                    f"{source.text}"
                    for index, source in enumerate(sources, start=1)
                )

    # On initialization and on mode change, this function set the system prompt
    # to the default prompt based on the mode (and user settings).
    @staticmethod
    def _get_default_system_prompt(mode: str) -> str:
        p = ""
        match mode:
            # For query chat mode, obtain default system prompt from settings
            case "Query Files":
                p = settings().ui.default_query_system_prompt
            # For chat mode, obtain default system prompt from settings
            case "LLM Chat (no context from files)":
                p = settings().ui.default_chat_system_prompt
            # For any other mode, clear the system prompt
            case _:
                p = ""
        return p

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    def _set_current_mode(self, mode: str) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        # Update placeholder and allow interaction if default system prompt is set
        if self._system_prompt:
            return gr.update(placeholder=self._system_prompt, interactive=True)
        # Update placeholder and disable interaction if no default system prompt is set
        else:
            return gr.update(placeholder=self._system_prompt, interactive=False)

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            files.add(file_name)
        return [[row] for row in files]

    def _upload_file(self, files: list[str]) -> None:
        logger.debug("Loading count=%s files", len(files))
        paths = [Path(file) for file in files]

        # remove all existing Documents with name identical to a new file upload:
        file_names = [path.name for path in paths]
        doc_ids_to_delete = []
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"] in file_names
            ):
                doc_ids_to_delete.append(ingested_document.doc_id)
        if len(doc_ids_to_delete) > 0:
            logger.info(
                "Uploading file(s) which were already ingested: %s document(s) will be replaced.",
                len(doc_ids_to_delete),
            )
            for doc_id in doc_ids_to_delete:
                self._ingest_service.delete(doc_id)

        self._ingest_service.bulk_ingest([(str(path.name), path) for path in paths])

    def _upload_URL_file(self, files: list[str]) -> None:
        root=tk.Tk()
        root.attributes("-topmost", True)
        root.geometry("450x190")
        root.title("URLs Form")
        root.resizable(False,False)
        root.config(background="#DEE7FB")
        main_title = Label(text="Introduce the URL(s) separated by commas", font=(14), fg="#DEE7FB", bg="#6165ED", width="300")
        main_title.pack()

        username_label  = Label(text="URL(s)", bg="#6165ED", fg="#DEE7FB", width="20")
        username_label.place(x=18, y=60)

        URLs = StringVar()

        URLs_entry = Entry(textvariable=URLs, width="40")

        URLs_entry.place(x=18, y=90)

        submit_btn = Button(root, text="Submit info", command=lambda: self.send_URLs_data(URLs, root), width="30", bg="#6165ED", fg="#DEE7FB")
        submit_btn.place(x=18, y=130)

        root.mainloop()

        
    def send_URLs_data(self, URLs_, root_) -> None:
            
            root_.destroy()

            URLs = URLs_.get()

            # Crear la carpeta URL_files si no existe
            carpeta_url_files = "URL_files"
            if not os.path.exists(carpeta_url_files):
                os.makedirs(carpeta_url_files)

            lista_urls = URLs.split(',')

            rutas_absolutas = []
    
            for url in lista_urls:
                # Eliminar espacios en blanco al principio y al final de la URL
                url = url.strip()

                try:
                    # Realizar la solicitud HTTP GET a la URL
                    response = requests.get(url)
                    
                    # Verificar si la solicitud fue exitosa
                    if response.status_code == 200:
                        # Parsear el contenido HTML de la página web
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extraer el texto sin las etiquetas HTML
                        texto_sin_etiquetas = soup.get_text()
                        
                        # Obtener el nombre del archivo (usando el nombre de dominio de la URL)
                        nombre_archivo = f"{url.replace('https://', '').replace('http://', '').replace('/', '_').replace(':', '_')}.txt"

                        # Ruta completa del archivo dentro de la carpeta URL_files
                        ruta_archivo = os.path.join(carpeta_url_files, nombre_archivo)
                        
                        # Guardar el contenido en un archivo de texto
                        with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
                            archivo.write(texto_sin_etiquetas)
                        
                        rutas_absolutas.append(os.path.abspath(ruta_archivo))

                        print(f"Contenido de la URL '{url}' guardado en el archivo: {ruta_archivo}")

                    else:
                        print(f"No se pudo obtener el contenido de la URL: {url}")
                except Exception as e:
                    print(f"Error al procesar la URL '{url}': {str(e)}")

            self._upload_file(rutas_absolutas)


    def _upload_IMG_file_(self, files: list[str]) -> None:
        
        rutas_absolutas = []

        carpeta_img_files = "IMG_files"
        if not os.path.exists(carpeta_img_files):
            os.makedirs(carpeta_img_files)

        path = Path(files[0])
        filename = path.stem+".txt"

        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
        response = captioner(files[0])

        info = "La imagen y documento " + filename + " contiene la siguiente informacion: La imagen muestra una escena vibrante de una playa concurrida. En primer plano, hay tres jóvenes sin camiseta, vestidos con pantalones cortos, que parecen estar jugando con una pelota. Uno de ellos tiene la pelota en su cabeza, mientras los otros dos observan, posiblemente esperando su turno para jugar. Detrás de ellos, la playa está llena de personas disfrutando del sol y el mar. Algunas personas están en el agua, mientras que otras se relajan en la arena o en tumbonas bajo sombrillas azules. Hay una mezcla de actividades: niños jugando, adultos conversando y bañistas nadando. Al fondo de la imagen, se puede ver un hotel de estilo mediterráneo, con paredes blancas y detalles arquitectónicos que incluyen balcones y cúpulas. Enfrente del hotel, ondean varias banderas, entre ellas la bandera de España y la bandera de la Unión Europea, lo que sugiere que esta playa podría estar ubicada en una región turística de España. El paisaje también incluye vegetación variada, con palmeras y otros árboles que añaden un toque tropical al entorno. La playa parece ser de arena fina y clara, con un mar tranquilo de aguas cristalinas que invita a los bañistas a refrescarse. En resumen, la imagen capta un día típico de verano en una playa europea concurrida, llena de vida y actividades recreativas, con un hotel de fondo que destaca por su arquitectura blanca y elegante."

        # Ruta completa del archivo dentro de la carpeta URL_files
        ruta_archivo = os.path.join(carpeta_img_files, filename)
                        
        # Guardar el contenido en un archivo de texto
        with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
            archivo.write(info)
                        
        rutas_absolutas.append(os.path.abspath(ruta_archivo))

        self._upload_file(rutas_absolutas)

    
    def _upload_IMG_file(self, files: list[str]) -> None:
        
        rutas_absolutas = []

        carpeta_img_files = "IMG_files"
        if not os.path.exists(carpeta_img_files):
            os.makedirs(carpeta_img_files)

        path = Path(files[0])
        filename = path.stem+".txt"

        resultado = subprocess.run(
            ["python", "C:/Users/julian/Desktop/ejemplo/vision.py"] + files,  # Comando a ejecutar
            capture_output=True,        # Capturar la salida del script
            text=True                   # Devolver la salida como texto (string)
        )

        print(resultado.stdout.strip())

        info = "La imagen y documento " + filename + " contiene la siguiente informacion: " + resultado.stdout.strip()
        # Ruta completa del archivo dentro de la carpeta URL_files
        ruta_archivo = os.path.join(carpeta_img_files, filename)
                        
        # Guardar el contenido en un archivo de texto
        with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
            archivo.write(info)
                        
        rutas_absolutas.append(os.path.abspath(ruta_archivo))

        self._upload_file(rutas_absolutas)


    def _upload_AUDIO_file(self, files: list[str]) -> None:
        
        rutas_absolutas = []

        carpeta_audio_files = "AUDIO_files"
        if not os.path.exists(carpeta_audio_files):
            os.makedirs(carpeta_audio_files)

        path = Path(files[0])
        filename = path.stem+".txt"
        
        whisper = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        )
        
        response = whisper(files[0], batch_size=8, generate_kwargs={"language": "es"})["text"]
        print(response)

        info = "El audio y documento " + filename + " contiene esta conversación: " + response

        # Ruta completa del archivo dentro de la carpeta URL_files
        ruta_archivo = os.path.join(carpeta_audio_files, filename)
                        
        # Guardar el contenido en un archivo de texto
        with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
            archivo.write(info)
                        
        rutas_absolutas.append(os.path.abspath(ruta_archivo))

        self._upload_file(rutas_absolutas)


    def actualizar_historial(self):
        i=0
        historial = []
        for message in all_messages_g:

            if i % 2 == 0:
                # Si es una pregunta, crear un nuevo diccionario de conversación
                new_historial = {
                    "pregunta": message.content,
                    "respuesta": ""
                }
            else:
                # Si es una respuesta, agregarla al diccionario de conversación previo
                new_historial["respuesta"] = message.content
                # Agregar la conversación completa al historial
                historial.append(new_historial)
        
            i += 1

        # Definir la ruta de la carpeta info_usuarios
        carpeta_info_usuarios = "info_usuarios"

        # Crear la carpeta si no existe
        if not os.path.exists(carpeta_info_usuarios):
            os.makedirs(carpeta_info_usuarios)

        # Definir la ruta de la carpeta historiales dentro de info_usuarios
        carpeta_historiales = os.path.join(carpeta_info_usuarios, "historiales")

        # Crear la carpeta historiales si no existe
        if not os.path.exists(carpeta_historiales):
            os.makedirs(carpeta_historiales)

        # Definir el nombre del archivo dentro de la carpeta historiales
        filename = os.path.join(carpeta_historiales, "historial_" + user_name + ".json")

        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                json.dump({"historiales": []}, f, indent=4)

        # Cargar los usuarios existentes del archivo JSON
        with open(filename, 'r') as f:
            data = json.load(f)

        # Obtener la fecha y hora actual
        fecha_hora_actual = datetime.datetime.now()

        # Formatear la fecha y hora según tus preferencias
        fecha_hora_formateada = fecha_hora_actual.strftime("%Y-%m-%d %H:%M:%S")

        # Agregar el nuevo usuario a la lista de usuarios
        data["historiales"].append(fecha_hora_formateada)
        data["historiales"].append(historial)

        # Guardar los datos actualizados en el archivo JSON
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

        print("Datos guardados correctamente.")

    def mostrar_historial(self):

                carpeta_info_usuarios = "info_usuarios"

                carpeta_historiales = os.path.join(carpeta_info_usuarios, "historiales")

                filename = os.path.join(carpeta_historiales, "historial_" + user_name + ".json")

                global historial_str
                
                with open(filename, "r") as archivo:
                    historial_str = archivo.read()

                data = json.loads(historial_str)

                output_lines = []

                if "historiales" in data and isinstance(data["historiales"], list):
                    historiales = data["historiales"]
                    
                    i = 0
                    while i < len(historiales):
                        fecha = historiales[i]
                        i += 1

                        if i < len(historiales) and isinstance(historiales[i], list):
                            qa_list = historiales[i]
                            i += 1

                            output_lines.append(fecha)

                            for qa in qa_list:
                                 
                                output_lines.append("")
                                pregunta = qa.get("pregunta", "")
                                respuesta = qa.get("respuesta", "")
                                output_lines.append(f"PREGUNTA: {pregunta}")
                                output_lines.append(f"RESPUESTA: {respuesta}")  

                            output_lines.append("")

                output_str = "\n".join(output_lines).strip()

                self.create_scrollable_window(output_str, "Historial")


    def create_scrollable_window(self, content, title):
        root = tk.Tk()
        root.attributes("-topmost", True)
        root.title(title)
        root.geometry("700x900")
        root.resizable(False, False)
        root.config(background="#6165ED")

        # Crear un marco para el contenido scrollable
        frame = Frame(root, bg="#DEE7FB")
        frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Crear un widget Text scrollable dentro del marco
        text = Text(frame, wrap="word", width=60, height=15, bg="#DEE7FB", fg="#6165ED", font=("Arial", 12))
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Agregar una barra de desplazamiento
        scrollbar = Scrollbar(frame, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)

        # Insertar el contenido en el widget Text
        text.insert(tk.END, content)

        root.mainloop()

    def guardar_historial(self):
        
        root = tk.Tk()
        root.withdraw()  # Ocultar la ventana principal, ya que solo queremos mostrar el mensaje

        # Asegurarse de que la ventana principal está en primer plano
        root.attributes("-topmost", True)

        messagebox.showinfo("Intrucciones", "Para poder guardar el historial, introducir /Guardar en el chat")

        # Quitar el atributo de estar en primer plano y cerrar la ventana principal
        root.attributes("-topmost", False)
        root.destroy()

    def _delete_all_files(self) -> Any:
        ingested_files = self._ingest_service.list_ingested()
        logger.debug("Deleting count=%s files", len(ingested_files))
        for ingested_document in ingested_files:
            self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _delete_selected_file(self) -> Any:
        logger.debug("Deleting selected %s", self._selected_filename)
        # Note: keep looping for pdf's (each page became a Document)
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"]
                == self._selected_filename
            ):
                self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _deselect_selected_file(self) -> Any:
        self._selected_filename = None
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> Any:
        self._selected_filename = select_data.value
        return [
            gr.components.Button(interactive=True),
            gr.components.Button(interactive=True),
            gr.components.Textbox(self._selected_filename),
        ]

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(),
            css=".logo { "
                    "display: flex;"
                    "background-color: #DEE7FB;" 
                    "height: 80px;"
                    "border-radius: 8px;"
                    "align-items: center;"
                    "justify-content: center;"
                    "}"
                    "body { background-color: #FFFFFF; }"  
                    ".small-label { font-size: 8px; color: #DEE7FB; }"  
                    ".logo img { height: 25% }"
                    ".button { "
                    "background-color: #6165ed;"  
                    "color: #FFFFFF;"  
                    "border: 1px solid #DEE7FB;"  
                    "border-radius: 4px;"
                    "}"
                    ".button:hover { "
                    "background-color: #3451B3;"  
                    "}"
                    ".contain { display: flex !important; flex-direction: column !important; }"
                    "#component-0, #component-3, #component-10, #component-8 { height: 100% !important; }"
                    "#chatbot { flex-grow: 1 !important; overflow: auto !important; background-color: #DEE7FB}"
                    "#col { height: calc(100vh - 112px - 16px) !important; }"
                    "#grlista { background-color: #DEE7FB; }"
                    "#grradio { background-color: #DEE7FB; color: #FFFFFF; }"
                    ".grrtb { background-color: #DEE7FB; }"
                    ".grTextbox {"
                    "background-color: #6165ed;"  
                    "border: 1px solid #DEE7FB;"  
                    "border-radius: 4px;"
                    "font-size: 16px;" 
                    "text-align: center;" 
                    "display: grid;" 
                    "align-items: center;" 
                    "padding: 0 10px;" 
                    "}",
        ) as blocks:
            with gr.Row():
                with gr.Column(scale=10):  # This column now uses 75% of the row
                    gr.HTML(f"<div class='logo'><img src={logo_svg} alt='PrivateGPT'></div>")
                with gr.Column(scale=2):  # This column now uses 25% of the row
                    
                    
                    login_button = gr.Button("Log In", size="sm", elem_classes="button")
                    register_button = gr.Button("Sign Up", size="sm", elem_classes="button")
                    logout_button = gr.Button("Log Out", size="sm", visible=False, elem_classes="button")
                    user_status = gr.Textbox(value="b", container=None, label=None, visible=False, elem_classes="grTextbox")
                    
                    
            def logout():
                global user_id
                user_id = "a"
                print("Sesion de usuario acabada")
                # Crear la ventana principal
                root = tk.Tk()
                root.withdraw()  # Ocultar la ventana principal, ya que solo queremos mostrar el mensaje

                # Asegurarse de que la ventana principal está en primer plano
                root.attributes("-topmost", True)

                messagebox.showinfo("Sesión finalizada", f"Sesión de {user_name} finalizada")

                # Quitar el atributo de estar en primer plano y cerrar la ventana principal
                root.attributes("-topmost", False)
                root.destroy()
                # Hacer los botones de login y register invisibles
                login_button_visibility = gr.update(visible=True)
                register_button_visibility = gr.update(visible=True)
                # Hacer los botones de guardar y mostrar visibles
                mostrar_hist_button_visibility = gr.update(visible=False)
                user_status_visibility = gr.update(visible=False)
                logout_button_visibility = gr.update(visible=False)
                            
                return login_button_visibility, register_button_visibility, mostrar_hist_button_visibility, user_status_visibility, logout_button_visibility


            def login_popup():
               
                    root=tk.Tk()
                    root.attributes("-topmost", True)
                    root.geometry("450x250")
                    root.title("Log in Form")
                    root.resizable(False,False)
                    root.config(background="#DEE7FB")
                    main_title = Label(text="Introduce your information", font=(14), fg="#DEE7FB", bg="#6165ED", width="300")
                    main_title.pack()

                    username_label  = Label(text="Username", bg="#6165ED", fg="#DEE7FB", width="20")
                    username_label.place(x=18, y=60)
                    password_label  = Label(text="Password", bg="#6165ED", fg="#DEE7FB", width="20")
                    password_label.place(x=18, y=120)

                    username = StringVar()
                    password = StringVar()

                    username_entry = Entry(textvariable=username, width="40")
                    password_entry = Entry(textvariable=password, width="40", show='*')

                    username_entry.place(x=18, y=90)
                    password_entry.place(x=18, y= 150)

                    submit_btn = Button(root, text="Submit info", command=lambda: send_log_data(username, password, root), width="30", bg="#6165ED", fg="#DEE7FB")
                    submit_btn.place(x=18, y=190)

                    root.mainloop()

                    # Hacer los botones de login y register invisibles
                    login_button_visibility = gr.update(visible=False)
                    register_button_visibility = gr.update(visible=False)
                    # Hacer los botones de guardar y mostrar visibles
                    mostrar_hist_button_visibility = gr.update(visible=True)
                    user_status_visibility = gr.update(value="                             "+user_name, visible=True)
                    logout_button_visibility = gr.update(visible=True)
                    
                    return login_button_visibility, register_button_visibility, mostrar_hist_button_visibility, user_status_visibility, logout_button_visibility

                

            def register_popup():
                
                    root=tk.Tk()
                    root.attributes("-topmost", True)
                    root.geometry("450x308")
                    root.title("Sign up Form")
                    root.resizable(False,False)
                    root.config(background="#DEE7FB")
                    main_title = Label(text="Introduce your information", font=(14), fg="#DEE7FB", bg="#6165ED", width="300")
                    main_title.pack()

                    username_label  = Label(text="Username", bg="#6165ED", fg="#DEE7FB", width="20")
                    username_label.place(x=18, y=60)
                    password_label  = Label(text="Password", bg="#6165ED", fg="#DEE7FB", width="20")
                    password_label.place(x=18, y=120)
                    password2_label  = Label(text="Repeat Password", bg="#6165ED", fg="#DEE7FB", width="20")
                    password2_label.place(x=18, y=180)

                    username = StringVar()
                    password = StringVar()
                    password2 = StringVar()

                    username_entry = Entry(textvariable=username, width="40")
                    password_entry = Entry(textvariable=password, width="40", show='*')
                    password2_entry = Entry(textvariable=password2, width="40", show='*')

                    username_entry.place(x=18, y=90)
                    password_entry.place(x=18, y= 150)
                    password2_entry.place(x=18, y= 210)

                    submit_btn = Button(root, text="Submit info", command=lambda: send_register_data(username, password, password2, root), width="30", bg="#6165ED", fg="#DEE7FB")
                    submit_btn.place(x=18, y=250)

                    root.mainloop()


            def send_log_data(username_, password_, root_):
                root_.attributes("-topmost", False)
                root_.destroy()

                global user_id
                global user_name

                username = username_.get()
                password = str(password_.get())

                # Ruta del archivo usuarios.json dentro de la carpeta info_usuarios
                ruta_archivo = os.path.join("info_usuarios", "usuarios.json")

                # Abrir el archivo y cargar los datos JSON
                with open(ruta_archivo, 'r') as f:
                    data = json.load(f)

                # Buscar un usuario con el mismo nombre de usuario y contraseña
                for user in data["usuarios"]:
                    if user["username"] == username and user["password"] == password:
                        user_id = user["id"]
                        user_name = user["username"]

                        print("Inicio de sesión exitoso.")
                        messagebox.showinfo(user_name,"Inicio de sesión exitoso")

                        # Hacer los botones de login y register invisibles
                        login_button_visibility = gr.update(visible=False)
                        register_button_visibility = gr.update(visible=False)
                        # Hacer los botones de guardar y mostrar visibles
                        mostrar_hist_button_visibility = gr.update(visible=True)
                        guardar_hist_button_visibility = gr.update(visible=True)
                        user_status_visibility = gr.update(value="                             "+user_name, visible=True)
                        logout_button_visibility = gr.update(visible=True)
                    
                        return login_button_visibility, register_button_visibility, mostrar_hist_button_visibility, guardar_hist_button_visibility, user_status_visibility, logout_button_visibility
                        

                messagebox.showerror("Error","Nombre de usuario o contraseña incorrectos.")


            def send_register_data(username, password, password2, root):
                root.attributes("-topmost", False)
                root.destroy()

                # Obtener los datos de las entradas
                username_value = username.get()
                password_value = password.get()
                password2_value = password2.get()

                # Verificar si las contraseñas coinciden
                if password_value != password2_value:
                    print("Las contraseñas no coinciden.")
                    return [None, None]  # No hacer nada si hay error

                # Crear un identificador único para el usuario
                user_id = str(uuid.uuid4())

                # Crear un diccionario con los datos del nuevo usuario
                new_user = {
                    "id": user_id,
                    "username": username_value,
                    "password": password_value
                }

                # Verificar si el archivo JSON de usuarios existe, si no, crearlo

                # Nombre de la carpeta donde queremos colocar el archivo
                carpeta_info_usuarios = "info_usuarios"

                # Verificar si la carpeta info_usuarios existe, si no, crearla
                if not os.path.exists(carpeta_info_usuarios):
                    os.makedirs(carpeta_info_usuarios)

                filename = os.path.join(carpeta_info_usuarios, "usuarios.json")

                if not os.path.exists(filename):
                    with open(filename, 'w') as f:
                        json.dump({"usuarios": []}, f, indent=4)

                # Cargar los usuarios existentes del archivo JSON
                with open(filename, 'r') as f:
                    data = json.load(f)

                # Agregar el nuevo usuario a la lista de usuarios
                data["usuarios"].append(new_user)

                # Guardar los datos actualizados en el archivo JSON
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)

                print("Datos guardados correctamente.")
                messagebox.showinfo(user_name,"Registro exitoso")

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    mode = gr.Radio(
                        MODES,
                        label="Mode",
                        value="Query Files",
                        elem_id="grradio"
                    )
                    upload_button = gr.components.UploadButton(
                        "Upload File",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                        elem_classes="button"
                    )
                    upload_URL_button = gr.Button(
                        "Upload URL(s)",
                        size="sm",
                        elem_classes="button"
                    )
                    upload_IMG_button = gr.components.UploadButton(
                        "Upload Image",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                        elem_classes="button"
                    )
                    upload_AUDIO_button = gr.components.UploadButton(
                        "Upload Audio",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                        elem_classes="button"
                    )
                    ingested_dataset = gr.List(
                        self._list_ingested_files,
                        headers=["File name"],
                        label="Ingested Files",
                        height=235,
                        interactive=False,
                        render=False,  # Rendered under the button
                        elem_id="grlista"
                    )
                    upload_button.upload(
                        self._upload_file,
                        inputs=upload_button,
                        outputs=ingested_dataset,
                    )
                    upload_URL_button.click(
                        self._upload_URL_file,
                        inputs=upload_URL_button,
                        outputs=ingested_dataset,
                    )
                    upload_IMG_button.upload(
                        self._upload_IMG_file,
                        inputs=upload_IMG_button,
                        outputs=ingested_dataset,
                    )
                    upload_AUDIO_button.upload(
                        self._upload_AUDIO_file,
                        inputs=upload_AUDIO_button,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.change(
                        self._list_ingested_files,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()
                    deselect_file_button = gr.components.Button(
                        "De-select selected file", size="sm", interactive=False, elem_classes="button"
                    )
                    selected_text = gr.components.Textbox(
                        "All files", label="Selected for Query or Deletion", max_lines=1, elem_classes="grrtb"
                    )
                    delete_file_button = gr.components.Button(
                        "🗑️ Delete selected file",
                        size="sm",
                        visible=settings().ui.delete_file_button_enabled,
                        interactive=False,
                        elem_classes="button"
                    )
                    delete_files_button = gr.components.Button(
                        "⚠️ Delete ALL files",
                        size="sm",
                        visible=settings().ui.delete_all_files_button_enabled,
                        elem_classes="button"
                    )
                    mostrar_hist_button = gr.Button(
                        "Mostrar Historial",
                        size="sm",
                        visible=False,
                        elem_classes="button"
                    )
                    login_button.click(
                        fn=login_popup,
                        inputs=[],
                        outputs=[login_button, register_button, mostrar_hist_button, 
                                user_status, logout_button]
                    )
                    register_button.click(
                        fn=register_popup,
                        inputs=[],
                        outputs=None
                    )
                    mostrar_hist_button.click(
                        fn=self.mostrar_historial,
                        inputs=[],
                        outputs=None
                    )

                    logout_button.click(
                        fn=logout,
                        inputs=[],
                        outputs=[login_button, register_button, mostrar_hist_button, user_status, logout_button]
                    )

                    deselect_file_button.click(
                        self._deselect_selected_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    ingested_dataset.select(
                        fn=self._selected_a_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_file_button.click(
                        self._delete_selected_file,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_files_button.click(
                        self._delete_all_files,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    system_prompt_input = gr.Textbox(
                        placeholder=self._system_prompt,
                        label="System Prompt",
                        lines=2,
                        interactive=True,
                        render=False,
                        elem_classes="grrtb"
                    )
                    # When mode changes, set default system prompt
                    mode.change(
                        self._set_current_mode, inputs=mode, outputs=system_prompt_input
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt_input.blur(
                        self._set_system_prompt,
                        inputs=system_prompt_input,
                    )

                    def get_model_label() -> str | None:
                        """Get model label from llm mode setting YAML.

                        Raises:
                            ValueError: If an invalid 'llm_mode' is encountered.

                        Returns:
                            str: The corresponding model label.
                        """
                        # Get model label from llm mode setting YAML
                        # Labels: local, openai, openailike, sagemaker, mock, ollama
                        config_settings = settings()
                        if config_settings is None:
                            raise ValueError("Settings are not configured.")

                        # Get llm_mode from settings
                        llm_mode = config_settings.llm.mode

                        # Mapping of 'llm_mode' to corresponding model labels
                        model_mapping = {
                            "llamacpp": config_settings.llamacpp.llm_hf_model_file,
                            "openai": config_settings.openai.model,
                            "openailike": config_settings.openai.model,
                            "sagemaker": config_settings.sagemaker.llm_endpoint_name,
                            "mock": llm_mode,
                            "ollama": config_settings.ollama.llm_model,
                            "gemini": config_settings.gemini.model,
                        }

                        if llm_mode not in model_mapping:
                            print(f"Invalid 'llm mode': {llm_mode}")
                            return None

                        return model_mapping[llm_mode]

                with gr.Column(scale=7, elem_id="col"):
                    # Determine the model label based on the value of PGPT_PROFILES
                    model_label = get_model_label()
                    if model_label is not None:
                        label_text = (
                            f"LLM: {settings().llm.mode} | Model: {model_label}"
                        )
                    else:
                        label_text = f"LLM: {settings().llm.mode}"

                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(
                            label=label_text,
                            show_copy_button=True,
                            elem_id="chatbot",
                            render=False,
                            avatar_images=(
                                None,
                                AVATAR_BOT,
                            ),
                        ),
                        additional_inputs=[mode, upload_button, system_prompt_input],
                    )
        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
