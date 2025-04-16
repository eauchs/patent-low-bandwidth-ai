# -*- coding: utf-8 -*-
import os
import logging
import base64
import json
import re
import requests # Pour appeler vlmserver pour le texte
import tempfile
import shutil
import sys
import time
from io import BytesIO
from urllib.parse import urljoin # Gardé
from pathlib import Path
import threading
from datetime import datetime, timezone
import uuid
from typing import List, Dict, Optional, Tuple, Any, Union

# --- Flask & Extensions ---
from flask import Flask, make_response, request, jsonify, Response
from flask_cors import CORS

# --- Machine Learning & NLP (Embedding, Reranker ET SmolDocling chargés localement) ---
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import chromadb

# --- Gestion PDF ---
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    logging.info("pdf2image trouvé. Traitement PDF local activé (si Poppler est installé).")
except ImportError:
    logging.warning("pdf2image non trouvé. Traitement PDF local désactivé.")
    PDF2IMAGE_AVAILABLE = False
    convert_from_path = None
except Exception as e:
    logging.warning(f"Erreur init pdf2image (Poppler?): {e}. Traitement PDF désactivé.")
    PDF2IMAGE_AVAILABLE = False
    convert_from_path = None

# --- MLX-VLM (pour SmolDocling DIRECTEMENT) & Docling Imports ---
try:
    import mlx.core as mx
    from mlx_vlm.utils import load, generate as mlx_generate, stream_generate as mlx_stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    # Docling pour le post-traitement SmolDocling si besoin
    from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
    MLX_VLM_AVAILABLE = True
    logging.info("Bibliothèques mlx_vlm et docling_core chargées (pour SmolDocling).")
except ImportError as e:
    logging.warning(f"Import mlx/vlm/docling échoué: {e}. Analyse VLM (SmolDocling) désactivée.")
    MLX_VLM_AVAILABLE = False
    mx = None
    load = lambda path, **kwargs: (None, None) # Stub
    mlx_generate = None
    mlx_stream_generate = None
    apply_chat_template = None
    load_config = lambda path: None
    DocTagsDocument = None
    DoclingDocument = None

# --- Autres ---
import wikipedia
from dotenv import load_dotenv
import traceback # Pour logging erreurs

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger('werkzeug').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
class AppConfig:
    """Classe pour encapsuler la configuration."""
    def __init__(self):
        load_dotenv()
        self.TEMP_FOLDER: str = tempfile.gettempdir()
        # ChromaDB
        self.CHROMA_DB_PATH: str = os.environ.get("CHROMA_DB_PATH", "chromadb_base")
        self.CHROMA_COLLECTION_NAME: str = os.environ.get("CHROMA_COLLECTION_NAME", "my_collection")
        self.CHROMA_SMS_HISTORY_COLLECTION_NAME: str = os.environ.get("CHROMA_SMS_HISTORY_COLLECTION_NAME", "sms_history_collection")
        # Modèles ML (Locaux)
        self.EMBEDDING_MODEL_NAME: str = os.environ.get("EMBEDDING_MODEL_NAME", "ibm-granite/granite-embedding-278m-multilingual")
        self.RERANKER_MODEL_NAME: str = os.environ.get("RERANKER_MODEL_NAME", "ibm-research/re2g-reranker-nq")
        # >>> Chemin vers SmolDocling pour l'extraction image/PDF locale <<<
        self.SMOL_VLM_MODEL_PATH: Optional[str] = os.environ.get("SMOL_VLM_MODEL_PATH", "ds4sd/SmolDocling-256M-preview-mlx-bf16") if MLX_VLM_AVAILABLE else None

        # >>> Configuration pour le LLM Texte (pointe vers vlmserver.py @ 5001) <<<
        self.TEXT_GENERATION_SERVER_ENDPOINT: str = os.environ.get("TEXT_GENERATION_SERVER_ENDPOINT", "http://localhost:5001/api/vlm_generate")
        self.REQUEST_TIMEOUT: int = int(os.environ.get("REQUEST_TIMEOUT", 1000)) # Timeout pour les appels au serveur externe

        # Paramètres VLM (SmolDocling via appel direct)
        self.SMOL_VLM_MAX_TOKENS: int = int(os.environ.get("SMOL_VLM_MAX_TOKENS", 1024))
        self.SMOL_VLM_TEMPERATURE: float = float(os.environ.get("SMOL_VLM_TEMPERATURE", 0.0))

        # Paramètres Wikipedia, RAG, PDF, Reranker (inchangés)
        self.WIKIPEDIA_LANG: str = os.environ.get("WIKIPEDIA_LANG", "fr")
        self.WIKIPEDIA_MAX_CHARS: int = int(os.environ.get("WIKIPEDIA_MAX_CHARS", 8000))
        self.RAG_N_RESULTS: int = int(os.environ.get("RAG_N_RESULTS", 15))
        self.RAG_MIN_TOKEN_COUNT: int = int(os.environ.get("RAG_MIN_TOKEN_COUNT", 10))
        self.RAG_TOP_K_RERANKED: int = int(os.environ.get("RAG_TOP_K_RERANKED", 10))
        self.CONTEXT_MAX_CHARS_DOCUMENT: int = int(os.environ.get("CONTEXT_MAX_CHARS_DOCUMENT", 6000)) # Max chars extrait du doc
        self.CONTEXT_MAX_CHARS_RAG: int = int(os.environ.get("CONTEXT_MAX_CHARS_RAG", 10000))
        self.CONTEXT_MAX_CHARS_WIKI: int = int(os.environ.get("CONTEXT_MAX_CHARS_WIKI", 4000))
        self.SMS_HISTORY_RAG_N_RESULTS: int = int(os.environ.get("SMS_HISTORY_RAG_N_RESULTS", 10))
        self.CONTEXT_MAX_CHARS_SMS_HISTORY: int = int(os.environ.get("CONTEXT_MAX_CHARS_SMS_HISTORY", 5000))
        self.PDF_MAX_PAGES: int = int(os.environ.get("PDF_MAX_PAGES", 10))
        self.PDF_DPI: int = int(os.environ.get("PDF_DPI", 200))
        self.PDF_CONVERT_TIMEOUT: int = int(os.environ.get("PDF_CONVERT_TIMEOUT", 120))
        self.RERANKER_MAX_LENGTH: int = 512

CONFIG = AppConfig()

# --- Globals / State ---
# >>> Variables pour SmolDocling chargé localement <<<
smol_model, smol_processor, smol_config = None, None, None
# Modèles Embedding/Reranker
granite_embedding_processor: Optional[SentenceTransformer] = None
reranker_tokenizer, reranker_model = None, None
# Collections ChromaDB
chroma_collection: Optional[chromadb.Collection] = None
chroma_sms_history_collection: Optional[chromadb.Collection] = None
# Statut initialisation
INITIALIZATION_SUCCESSFUL = False

# --- Custom Exceptions --- (Gardez vos exceptions)
class InitializationError(Exception): pass
class VLMProcessingError(Exception): pass
class RAGError(Exception): pass
class LLMError(Exception): pass # Pour erreurs serveur texte externe
class FileProcessingError(Exception): pass

# --- Regex ---
WORD_TOKEN_REGEX = re.compile(r'\b\w+\b')

# --- Model Initialization ---
def initialize_models() -> bool:
    """Charge les modèles locaux (Embedding, Reranker, SmolDocling) et init ChromaDB."""
    global smol_model, smol_processor, smol_config, granite_embedding_processor
    global reranker_tokenizer, reranker_model
    global chroma_collection, chroma_sms_history_collection
    global INITIALIZATION_SUCCESSFUL
    global MLX_VLM_AVAILABLE # Utilise la variable globale définie lors des imports VLM

    success = True
    logger.info("Début initialisation composants locaux (Embedding, Reranker, SmolDocling, ChromaDB)...")

    # 1. VLM (SmolDocling pour extraction image/PDF locale)
    if MLX_VLM_AVAILABLE and CONFIG.SMOL_VLM_MODEL_PATH:
        try:
            logger.info(f"VLM (SmolDocling): Chargement depuis {CONFIG.SMOL_VLM_MODEL_PATH}...")
            smol_model, smol_processor = load(CONFIG.SMOL_VLM_MODEL_PATH, trust_remote_code=True)
            smol_config = load_config(CONFIG.SMOL_VLM_MODEL_PATH, trust_remote_code=True)
            if smol_model is None or smol_processor is None or smol_config is None:
                raise ValueError("load ou load_config a retourné None pour SmolDocling")
            logger.info(f"VLM (SmolDocling): Modèle '{CONFIG.SMOL_VLM_MODEL_PATH}' chargé localement.")
        except Exception as e:
            logger.error(f"Échec chargement VLM (SmolDocling) local: {e}", exc_info=True)
            MLX_VLM_AVAILABLE = False
            logger.warning("VLM (SmolDocling): Fonctionnalité extraction image/PDF désactivée.")
    else:
         logger.warning("VLM (SmolDocling): Non configuré ou bibliothèques manquantes. Extraction image/PDF désactivée.")
         MLX_VLM_AVAILABLE = False # Assurer que c'est False si non chargé

    # 2. Embedding Model (Critique)
    try:
        logger.debug(f"Embedding: Chargement {CONFIG.EMBEDDING_MODEL_NAME}...")
        granite_embedding_processor = SentenceTransformer(CONFIG.EMBEDDING_MODEL_NAME)
        logger.info(f"Embedding: Modèle '{CONFIG.EMBEDDING_MODEL_NAME}' chargé.")
    except Exception as e:
        logger.critical(f"CRITIQUE: Échec chargement Embedding Model: {e}", exc_info=True)
        success = False

    # 3. Reranker Model (Critique)
    try:
        logger.debug(f"Reranker: Chargement {CONFIG.RERANKER_MODEL_NAME}...")
        reranker_tokenizer = AutoTokenizer.from_pretrained(CONFIG.RERANKER_MODEL_NAME)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(CONFIG.RERANKER_MODEL_NAME).eval()
        logger.info(f"Reranker: Modèle '{CONFIG.RERANKER_MODEL_NAME}' chargé.")
    except Exception as e:
        logger.critical(f"CRITIQUE: Échec chargement Reranker Model: {e}", exc_info=True)
        success = False

    # 4. ChromaDB Client & Collections (Critique)
    if success:
        try:
            logger.debug(f"ChromaDB: Connexion client persistant à {CONFIG.CHROMA_DB_PATH}")
            chroma_client = chromadb.PersistentClient(path=CONFIG.CHROMA_DB_PATH)
            logger.debug(f"ChromaDB: Get/Create collection RAG Docs '{CONFIG.CHROMA_COLLECTION_NAME}'")
            chroma_collection = chroma_client.get_or_create_collection(CONFIG.CHROMA_COLLECTION_NAME)
            logger.info(f"ChromaDB: Collection RAG Documents '{chroma_collection.name}' prête.")
            logger.debug(f"ChromaDB: Get/Create collection SMS History '{CONFIG.CHROMA_SMS_HISTORY_COLLECTION_NAME}'")
            chroma_sms_history_collection = chroma_client.get_or_create_collection(CONFIG.CHROMA_SMS_HISTORY_COLLECTION_NAME)
            logger.info(f"ChromaDB: Collection Historique SMS '{chroma_sms_history_collection.name}' prête.")
        except Exception as e:
            logger.critical(f"CRITIQUE: Échec connexion/création collection ChromaDB: {e}", exc_info=True)
            success = False
    else:
        logger.warning("Initialisation ChromaDB annulée car composants critiques manquants.")

    # 5. Vérifier connexion au serveur Texte externe (vlmserver.py @ 5001)
    try:
         logger.debug(f"Test connexion serveur Texte externe à {CONFIG.TEXT_GENERATION_SERVER_ENDPOINT}...")
         response = requests.post(CONFIG.TEXT_GENERATION_SERVER_ENDPOINT, timeout=5, json={}) # Envoyer JSON vide
         # Le serveur VLM modifié doit retourner 400 si 'prompt' manque, mais pas une erreur réseau
         if response.status_code == 400:
             logger.info(f"Serveur Texte externe semble répondre (status: {response.status_code}).")
         elif response.status_code == 503:
              logger.warning(f"Serveur Texte externe a répondu 503, modèle peut-être non prêt.")
         else:
             logger.warning(f"Réponse inattendue ({response.status_code}) du serveur Texte externe lors du test.")
             # On ne bloque pas le démarrage pour ça
    except requests.exceptions.ConnectionError:
         logger.error(f"ALERTE: Impossible de joindre le serveur Texte externe à {CONFIG.TEXT_GENERATION_SERVER_ENDPOINT}. La génération de texte échouera.")
         # Ne pas marquer success=False ici, l'app peut démarrer mais sera limitée.
    except Exception as e:
         logger.warning(f"Erreur lors du test de connexion au serveur Texte externe: {e}")

    INITIALIZATION_SUCCESSFUL = success
    logger.info(f"Initialisation terminée. Succès des composants critiques locaux: {success}")
    if not MLX_VLM_AVAILABLE: logger.warning("Fonctionnalité extraction Image/PDF (SmolDocling) non disponible.")
    if not PDF2IMAGE_AVAILABLE and MLX_VLM_AVAILABLE: logger.warning("Fonctionnalité extraction PDF désactivée (pdf2image/Poppler manquant).")

    return success


# --- Utility Functions ---
# count_tokens, get_wikipedia_context, rerank_documents_with_re2g, get_relevant_context_chromadb_docs
# (Inchangées, assurez-vous qu'elles sont présentes)
def count_tokens(text: str) -> int:
    """Compte les mots simples dans un texte."""
    if not text: return 0
    return len(WORD_TOKEN_REGEX.findall(text))

def get_wikipedia_context(query: str, lang: str = CONFIG.WIKIPEDIA_LANG, max_chars: int = CONFIG.WIKIPEDIA_MAX_CHARS) -> str:
    """Récupère un résumé de page Wikipedia."""
    logger.debug(f"Recherche Wikipedia pour: '{query}'")
    content = ""
    try:
        wikipedia.set_lang(lang)
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            logger.warning(f"Wikipedia: Aucun résultat trouvé pour '{query}'.")
            return ""

        page = None
        title_attempt = search_results[0]
        try:
            page = wikipedia.page(title=title_attempt, auto_suggest=False, redirect=True, preload=True)
            logger.info(f"Wikipedia: Contexte trouvé pour '{page.title}'.")
            content = page.content
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"Wikipedia: Page de désambiguïsation pour '{title_attempt}'. Options: {e.options}")
            if e.options:
                try:
                    page = wikipedia.page(title=e.options[0], auto_suggest=False, redirect=True, preload=True)
                    logger.info(f"Wikipedia: Contexte trouvé via désambiguïsation pour '{page.title}'.")
                    content = page.content
                except wikipedia.exceptions.PageError:
                     logger.warning(f"Wikipedia: Page '{e.options[0]}' non trouvée après désambiguïsation.")
                except Exception as inner_e:
                     logger.error(f"Wikipedia: Erreur lors de la récupération après désambiguïsation: {inner_e}")
        except wikipedia.exceptions.PageError:
            logger.warning(f"Wikipedia: Page '{title_attempt}' non trouvée.")
        except Exception as e:
             logger.error(f"Wikipedia: Erreur inattendue lors de la récupération de page '{title_attempt}': {e}", exc_info=True)

        if content:
            content = content[:max_chars].strip()
            logger.debug(f"Wikipedia: Contenu tronqué à {len(content)} caractères.")
        return content

    except Exception as e:
        logger.error(f"Wikipedia: Erreur générale lors de la recherche/récupération: {e}", exc_info=True)
        return ""

def rerank_documents_with_re2g(query: str, documents: List[str]) -> List[str]:
    """Re-classe les documents en utilisant le modèle RE2G."""
    if not documents or not query or not reranker_model or not reranker_tokenizer:
        logger.warning("Re-ranking annulé (modèle non dispo, query/docs vides?).")
        return documents
    if len(documents) == 1:
        logger.debug("Re-ranking : un seul document, pas de re-classement nécessaire.")
        return documents

    logger.debug(f"Début re-ranking de {len(documents)} documents pour query: '{query[:50]}...'")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        reranker_model.to(device)

        inputs = [f"{query} [SEP] {doc}" for doc in documents]
        encoded_inputs = reranker_tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt", max_length=CONFIG.RERANKER_MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            outputs = reranker_model(**encoded_inputs)
            scores = outputs.logits.squeeze(-1).cpu()

        scored_docs = sorted(zip(scores.tolist(), documents), reverse=True, key=lambda x: x[0])
        ranked_documents = [doc for score, doc in scored_docs]

        logger.info(f"Re-ranking terminé. {len(ranked_documents)} documents re-classés.")
        return ranked_documents

    except Exception as e:
        logger.error(f"Erreur durant le re-ranking: {e}", exc_info=True)
        return documents

def get_relevant_context_chromadb_docs(query: str) -> str:
    """Récupère et re-classe le contexte pertinent depuis ChromaDB (collection DOCUMENTS)."""
    if not granite_embedding_processor:
        logger.error("RAG Documents impossible: Embedding non initialisé.")
        raise RAGError("Composant Embedding non disponible.")
    if not chroma_collection:
        logger.error("RAG Documents impossible: Collection ChromaDB non disponible.")
        raise RAGError("Collection RAG Documents non disponible.")

    logger.debug(f"get_relevant_context_chromadb_docs: Début recherche pour query='{query[:50]}...'")
    try:
        logger.info(f"get_relevant_context_chromadb_docs: Interrogation collection '{chroma_collection.name}'")
        query_embedding = granite_embedding_processor.encode(query, convert_to_tensor=False).tolist()
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=CONFIG.RAG_N_RESULTS,
            include=["documents"]
        )

        context_documents = results["documents"][0] if results and results.get("documents") and results["documents"] else []
        logger.debug(f"get_relevant_context_chromadb_docs: Trouvé {len(context_documents)} documents bruts.")

        filtered_documents = [doc for doc in context_documents if count_tokens(doc) >= CONFIG.RAG_MIN_TOKEN_COUNT]
        logger.debug(f"get_relevant_context_chromadb_docs: {len(filtered_documents)} documents après filtrage token (min {CONFIG.RAG_MIN_TOKEN_COUNT}).")
        if not filtered_documents:
            logger.warning(f"RAG Documents: Aucun document trouvé après filtrage.")
            return ""

        ranked_documents = rerank_documents_with_re2g(query, filtered_documents)
        top_k_docs = ranked_documents[:CONFIG.RAG_TOP_K_RERANKED]
        logger.debug(f"get_relevant_context_chromadb_docs: {len(top_k_docs)} documents après reranking (top {CONFIG.RAG_TOP_K_RERANKED}).")

        context = "\n\n---\n\n".join(top_k_docs)
        context = context[:CONFIG.CONTEXT_MAX_CHARS_RAG]

        logger.info(f"get_relevant_context_chromadb_docs: Contexte final généré (longueur={len(context)}).")
        return context

    except Exception as e:
        logger.error(f"Erreur durant le processus RAG Documents: {e}", exc_info=True)
        raise RAGError(f"Erreur interne RAG Documents: {e}") from e

# --- Fonctions d'extraction de texte (Image/PDF utilisent SmolDocling LOCALEMENT) ---

# >>> VERSION ORIGINALE qui utilise SmolDocling chargé localement <<<
def extract_text_from_image_mlx_vlm(image_input: Union[str, Image.Image]) -> str:
    """
    Extrait le texte (Markdown) d'une image via SmolDocling local.
    Utilise les variables globales smol_model, smol_processor, smol_config.
    """
    global smol_model, smol_processor, smol_config # Important

    if not MLX_VLM_AVAILABLE or not smol_model or not smol_processor or not smol_config:
        message = "Extraction VLM (SmolDocling) impossible: Composants locaux non disponibles/chargés."
        logger.error(message)
        return f"[{message}]" # Retourne erreur plutôt que de planter

    # Le reste de la fonction est identique à la version originale que vous aviez
    # qui préparait l'image, appelait apply_chat_template et mlx_stream_generate,
    # puis faisait le post-traitement Docling.
    temp_image_path: Optional[str] = None
    pil_image: Optional[Image.Image] = None
    image_path_for_stream: Optional[str] = None
    final_output = "[VLM SmolDocling: Erreur inconnue]"

    try:
        if isinstance(image_input, str) and Path(image_input).is_file():
            image_path_for_stream = image_input
            try:
                pil_image = Image.open(image_input).convert("RGB")
            except Exception as img_err:
                raise VLMProcessingError(f"Impossible d'ouvrir l'image: {image_input} - {img_err}")
            logger.debug(f"VLM (SmolDocling): Traitement image depuis chemin: {os.path.basename(image_path_for_stream)}")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=CONFIG.TEMP_FOLDER) as tmp:
                try:
                    pil_image.save(tmp.name, format="PNG")
                    temp_image_path = tmp.name
                    image_path_for_stream = temp_image_path
                    logger.debug(f"VLM (SmolDocling): Image PIL sauvegardée temporairement: {os.path.basename(temp_image_path)}")
                except Exception as save_err:
                     raise VLMProcessingError(f"Impossible de sauvegarder l'image PIL temporairement: {save_err}")
        else:
            raise ValueError("Entrée invalide pour VLM: Doit être chemin de fichier ou objet PIL.Image.")

        if not pil_image or not image_path_for_stream:
             raise VLMProcessingError("Préparation de l'image échouée.")

        # Prompt spécifique pour SmolDocling
        prompt = "Convert this page to docling."
        formatted_prompt = apply_chat_template(smol_processor, smol_config, prompt, num_images=1)

        logger.debug(f"VLM (SmolDocling): Appel mlx_stream_generate local pour: {os.path.basename(image_path_for_stream)}")
        raw_model_output_chunks = []
        if mlx_stream_generate:
            try:
                for token in mlx_stream_generate(
                    smol_model, smol_processor, formatted_prompt, [image_path_for_stream],
                    max_tokens=CONFIG.SMOL_VLM_MAX_TOKENS, temp=CONFIG.SMOL_VLM_TEMPERATURE, verbose=False
                ):
                    raw_model_output_chunks.append(token.text) # Assumer structure correcte
                raw_model_output = "".join(raw_model_output_chunks)
                logger.info(f"VLM (SmolDocling): Sortie brute locale reçue ({len(raw_model_output)} chars).")
                if not raw_model_output.strip():
                    logger.warning("VLM (SmolDocling): Sortie brute modèle locale vide.")
                    final_output = "[VLM SmolDocling: Aucune sortie du modèle]"
                else:
                     # --- Post-traitement Docling (si utilisé) ---
                     if DocTagsDocument and DoclingDocument: # Vérifier si classes dispo
                         try:
                             logger.debug("VLM (SmolDocling): Début post-traitement Docling.")
                             doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([raw_model_output], [pil_image])
                             doc = DoclingDocument(name="ExtractedDocument")
                             doc.load_from_doctags(doctags_doc)
                             markdown_output = doc.export_to_markdown()
                             logger.info(f"VLM (SmolDocling): Post-traitement Docling OK -> Markdown ({len(markdown_output)} chars).")
                             final_output = markdown_output if markdown_output.strip() else "[VLM SmolDocling: Contenu vide après post-traitement]"
                         except Exception as post_err:
                             logger.error(f"VLM (SmolDocling): Erreur post-traitement Docling: {post_err}", exc_info=True)
                             final_output = f"[Erreur post-traitement Docling VLM: {type(post_err).__name__}]"
                     else:
                          logger.warning("Classes Docling non disponibles, retour de la sortie brute.")
                          final_output = raw_model_output # Retourner brut si Docling non dispo
            except Exception as gen_err:
                 logger.error(f"VLM (SmolDocling): Erreur durant mlx_stream_generate local: {gen_err}", exc_info=True)
                 final_output = f"[Erreur génération VLM SmolDocling: {type(gen_err).__name__}]"
        else:
             logger.warning("mlx_stream_generate non trouvé, VLM SmolDocling impossible.")
             final_output = "[VLM SmolDocling: Fonction stream_generate non disponible]"

    except Exception as e:
        logger.error(f"VLM (SmolDocling): Erreur générale extraction image locale: {e}", exc_info=True)
        final_output = f"[Erreur VLM SmolDocling: {type(e).__name__}]"

    finally:
        if temp_image_path and Path(temp_image_path).exists():
            try:
                os.remove(temp_image_path)
                logger.debug(f"VLM (SmolDocling): Fichier image temporaire supprimé: {temp_image_path}")
            except OSError as del_err:
                logger.warning(f"VLM (SmolDocling): Échec suppression fichier image temp {temp_image_path}: {del_err}")
        return final_output

# >>> Cette fonction utilise la précédente (SmolDocling local) <<<
def extract_text_from_pdf(file_path: str) -> str:
    """Extrait le texte d'un PDF en utilisant pdf2image et SmolDocling local."""
    if not PDF2IMAGE_AVAILABLE:
         message = "Traitement PDF impossible: pdf2image/Poppler non disponible."
         logger.error(message)
         return f"[{message}]"
    if not MLX_VLM_AVAILABLE: # Vérifie si SmolDocling est chargé
         message = "Traitement PDF impossible: VLM (SmolDocling local) non disponible."
         logger.error(message)
         return f"[{message}]"

    extracted_texts: List[str] = []
    logger.info(f"PDF: Début traitement {file_path} via SmolDocling local")

    try:
        images_pil = convert_from_path(
            file_path, dpi=CONFIG.PDF_DPI, fmt='png',
            thread_count=os.cpu_count() or 1, timeout=CONFIG.PDF_CONVERT_TIMEOUT, last_page=CONFIG.PDF_MAX_PAGES
        )
        logger.info(f"PDF: {len(images_pil)} pages converties en images PIL.")
        if not images_pil: return "[PDF: Aucune page traitable]"

        for i, img_pil in enumerate(images_pil):
            page_num = i + 1
            try:
                logger.debug(f"PDF: Extraction texte SmolDocling local pour page {page_num}...")
                # Appelle la fonction qui utilise SmolDocling directement
                page_text = extract_text_from_image_mlx_vlm(img_pil)

                if page_text and not page_text.startswith(("[Erreur", "[VLM SmolDocling:")):
                    extracted_texts.append(f"\n--- Page {page_num} ---\n{page_text}")
                    logger.debug(f"PDF: Texte extrait localement pour page {page_num}.")
                else:
                    logger.warning(f"PDF: Erreur ou contenu vide retourné par SmolDocling local pour page {page_num}: {page_text}")
                    extracted_texts.append(f"\n--- Page {page_num} ---\n{page_text}")
            except Exception as page_err:
                 logger.error(f"PDF: Erreur inattendue traitement page {page_num} avec SmolDocling: {page_err}", exc_info=True)
                 extracted_texts.append(f"\n--- Page {page_num} ---\n[Erreur interne traitement page {page_num}]")
            finally:
                 try: img_pil.close()
                 except Exception: pass

        return "\n".join(extracted_texts).strip() if extracted_texts else "[PDF: Aucun contenu extrait localement]"

    except Exception as pdf_err:
        logger.error(f"PDF: Erreur lors de la conversion PDF en images: {pdf_err}", exc_info=True)
        return f"[Erreur Conversion PDF: {type(pdf_err).__name__}]"

# >>> Inchangée <<<
def extract_text_from_simple_file(file_path: str) -> str:
    """Extrait le texte brut d'un fichier simple."""
    # (Code identique à la version précédente)
    logger.info(f"TEXT: Lecture du fichier: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Fichier simple non trouvé: {file_path}")
        return f"[Erreur: Fichier non trouvé {Path(file_path).name}]"
    except Exception as txt_err:
        logger.error(f"Erreur lecture fichier texte {file_path}: {txt_err}", exc_info=True)
        return f"[Erreur lecture fichier texte: {type(txt_err).__name__}]"

# >>> Utilise les fonctions d'extraction ci-dessus <<<
def extract_text_from_file(file_path: str) -> str:
    """Extrait le texte d'un fichier (PDF/Image via SmolDocling local, Texte direct)."""
    file_path_obj = Path(file_path)
    if not file_path_obj.is_file():
        logger.error(f"Chemin invalide pour extraction: {file_path}")
        return "[Erreur: Chemin de fichier invalide]"

    ext = file_path_obj.suffix.lower()
    logger.info(f"Extraction: Tentative traitement {file_path_obj.name} (type: {ext})")

    try:
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']:
            # Appelle la fonction qui utilise SmolDocling localement
            return extract_text_from_image_mlx_vlm(file_path)
        elif ext == '.pdf':
            # Appelle la fonction qui utilise SmolDocling localement pour chaque page
            return extract_text_from_pdf(file_path)
        elif ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
            return extract_text_from_simple_file(file_path)
        else:
            logger.warning(f"Extraction non supportée pour type fichier: {ext}")
            return f"[Type de fichier non supporté: {ext}]"
    except Exception as e:
        logger.error(f"Erreur inattendue extraction {file_path_obj.name}: {e}", exc_info=True)
        return f"[Erreur inattendue extraction fichier: {type(e).__name__}]"


# --- Fonctions de gestion de l'historique SMS avec Embeddings ---
# (Inchangées, assurez-vous qu'elles sont présentes)
def add_sms_to_history(sender: str, role: str, content: str):
    # (Code identique)
    if not granite_embedding_processor: logger.error("SMS History Add impossible: Embedding non initialisé."); return
    if not chroma_sms_history_collection: logger.error("SMS History Add impossible: Collection ChromaDB SMS non initialisée."); return
    try:
        timestamp_iso = datetime.now(timezone.utc).isoformat()
        message_id = f"sms_{sender}_{uuid.uuid4()}"
        embedding = granite_embedding_processor.encode(content, convert_to_tensor=False).tolist()
        chroma_sms_history_collection.add(
            documents=[content], embeddings=[embedding],
            metadatas=[{"sender_id": sender, "role": role, "timestamp": timestamp_iso}], ids=[message_id]
        )
        logger.debug(f"Message SMS '{role}' pour {sender} ajouté (ID: {message_id}).")
    except Exception as e: logger.error(f"Erreur ajout message SMS à ChromaDB pour {sender}: {e}", exc_info=True)

def get_relevant_sms_history(sender: str, query_message: str, n_results: int) -> List[Dict[str, Any]]:
    # (Code identique)
    relevant_history = []
    if not granite_embedding_processor: logger.error("SMS History Get impossible: Embedding non initialisé."); return relevant_history
    if not chroma_sms_history_collection: logger.error("SMS History Get impossible: Collection ChromaDB SMS non initialisée."); return relevant_history
    logger.debug(f"get_relevant_sms_history: Recherche pour sender='{sender}', query='{query_message[:50]}...'")
    try:
        logger.info(f"get_relevant_sms_history: Interrogation collection '{chroma_sms_history_collection.name}'")
        query_embedding = granite_embedding_processor.encode(query_message, convert_to_tensor=False).tolist()
        where_filter = {"sender_id": sender}
        results = chroma_sms_history_collection.query(
            query_embeddings=[query_embedding], n_results=n_results, where=where_filter, include=["documents", "metadatas"]
        )
        if results and results.get("ids") and results["ids"][0]:
            retrieved_docs = results["documents"][0]; retrieved_metadatas = results["metadatas"][0]
            for doc, meta in zip(retrieved_docs, retrieved_metadatas):
                relevant_history.append({"role": meta.get("role", "unknown"), "content": doc, "timestamp": meta.get("timestamp", "")})
            relevant_history.sort(key=lambda x: x.get("timestamp", ""))
            logger.info(f"get_relevant_sms_history: {len(relevant_history)} messages pertinents trouvés pour {sender}.")
        else: logger.info(f"get_relevant_sms_history: Aucun historique pertinent trouvé pour {sender}.")
    except Exception as e: logger.error(f"Erreur récupération historique SMS pertinent pour {sender}: {e}", exc_info=True)
    return relevant_history

def format_history_context(history: List[Dict[str, Any]], max_chars: int) -> str:
    # (Code identique)
    context_str = "Historique pertinent:\n"; total_len = len(context_str); added_messages = 0
    for msg in reversed(history): # Du plus récent au plus ancien
        line = f"{msg.get('role', '?')}: {msg.get('content', '')}\n"
        if total_len + len(line) <= max_chars:
            context_str = line + context_str # Ajoute au début
            total_len += len(line); added_messages += 1
        else: logger.debug(f"format_history_context: Limite ({max_chars}) atteinte après {added_messages} messages."); break
    if added_messages == 0: return ""
    final_context = "Historique pertinent:\n" + context_str.replace("Historique pertinent:\n", "").strip() + "\n---"
    logger.debug(f"format_history_context: Contexte formaté avec {added_messages} messages (longueur={len(final_context)}).")
    return final_context.strip()

# --- LLM Interaction (MODIFIÉE pour appeler vlmserver.py @ 5001 pour le TEXTE) ---
def generate_llm_response( # Renommée pour clarté
    prompt_user: str,
    conversation_history: List[Dict[str, Any]], # Non utilisé directement dans cette version
    context_external: Optional[str],
    temperature: float = 0.7,
    max_tokens: int = 512
) -> str:
    """
    Génère une réponse texte en appelant le serveur VLM externe (configuré pour texte).
    Construit un prompt unique à partir du contexte externe et du prompt utilisateur.
    """
    text_server_endpoint = CONFIG.TEXT_GENERATION_SERVER_ENDPOINT # Pointe vers vlmserver @ 5001
    headers = {"Content-Type": "application/json"}

    # Construction du Prompt unique envoyé au serveur externe
    system_prompt = "Tu es un assistant IA utile et concis."
    full_prompt_parts = [system_prompt]
    if context_external:
        full_prompt_parts.append("\n--- Contexte Pertinent ---")
        full_prompt_parts.append(context_external)
        full_prompt_parts.append("--- Fin Contexte ---")
    full_prompt_parts.append("\n--- Question/Demande ---")
    full_prompt_parts.append(prompt_user)
    final_prompt = "\n\n".join(full_prompt_parts).strip()

    logger.debug(f"Prompt final envoyé au serveur Texte externe (longueur={len(final_prompt)}):\n{final_prompt[:500]}...")

    # Payload pour le serveur VLM (utilisé pour texte ici, SANS image)
    payload = {
        "prompt": final_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    logger.info(f"Appel API serveur Texte externe vers {text_server_endpoint}...")
    response = None
    try:
        response = requests.post(
            text_server_endpoint, headers=headers, json=payload, timeout=CONFIG.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        response_json = response.json()
        logger.debug(f"Réponse JSON brute reçue du serveur Texte externe.")

        if response_json.get("response"):
            content = response_json["response"]
            logger.info(f"Génération Texte externe réussie.")
            return content.strip()
        else:
            error_detail = response_json.get("error", "Format de réponse inconnu")
            logger.error(f"Réponse invalide du serveur Texte externe: {error_detail}")
            raise LLMError(f"Réponse invalide du serveur Texte externe: {error_detail}")

    # --- Gestion des erreurs (identique) ---
    except requests.exceptions.HTTPError as e:
        error_text = f"Erreur HTTP {e.response.status_code}"
        try: error_text += f" - {e.response.json()}"
        except json.JSONDecodeError: error_text += f" - {e.response.text[:500]}"
        logger.error(f"Erreur appel serveur Texte externe: {error_text}", exc_info=False)
        raise LLMError(f"Erreur API serveur Texte externe ({e.response.status_code})") from e
    except requests.exceptions.Timeout:
        logger.error(f"Timeout ({CONFIG.REQUEST_TIMEOUT}s) lors de l'appel serveur Texte externe.")
        raise LLMError("Timeout API serveur Texte externe.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur connexion/requête serveur Texte externe API: {e}", exc_info=True)
        raise LLMError(f"Erreur réseau ou connexion serveur Texte externe: {e}") from e
    except json.JSONDecodeError:
        error_text = response.text[:500] if response else "N/A"
        logger.error(f"Réponse API serveur Texte externe non-JSON: {error_text}")
        raise LLMError("Réponse invalide (non-JSON) de l'API serveur Texte externe.")
    except Exception as e:
        logger.error(f"Erreur inattendue durant la génération Texte externe: {e}", exc_info=True)
        raise LLMError(f"Erreur interne inattendue serveur Texte externe: {e}") from e
    finally:
        if response: response.close()

# --- Flask App and Routes ---
app = Flask(__name__)
# Utiliser un nom différent si ce fichier coexiste avec l'autre
# app = Flask("main_app_hybrid")
CORS(app, resources={r"/api/*": {"origins": "*"}})

def make_error_response(message: str, status_code: int) -> Response:
    """Crée une réponse JSON standard pour les erreurs."""
    key = "reply" if request.path == '/api/sms_query' else "error"
    logger.warning(f"Réponse Erreur ({status_code}) pour {request.path}: {message}")
    return make_response(jsonify({key: f"[Erreur: {message}]"}), status_code)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint pour vérifier la disponibilité et l'initialisation."""
    status_details = {
        "status": "OK" if INITIALIZATION_SUCCESSFUL else "Error",
        "message": "" if INITIALIZATION_SUCCESSFUL else "Initialisation composants locaux critiques échouée.",
        "critical_components_initialized": INITIALIZATION_SUCCESSFUL,
        "embedding_model_loaded": granite_embedding_processor is not None,
        "reranker_model_loaded": reranker_model is not None,
        "smoldocling_vlm_available": MLX_VLM_AVAILABLE, # Indique si SmolDocling est prêt
        "chromadb_doc_collection_loaded": chroma_collection is not None,
        "chromadb_sms_collection_loaded": chroma_sms_history_collection is not None,
        "pdf_processing_available": PDF2IMAGE_AVAILABLE,
        "text_generation_server_target": CONFIG.TEXT_GENERATION_SERVER_ENDPOINT
    }
    status_code = 200 if INITIALIZATION_SUCCESSFUL else 503
    return jsonify(status_details), status_code

@app.route('/api/generate', methods=['POST', 'OPTIONS'])
def handle_generate():
    """Route principale pour génération avec contexte RAG / Wiki / Fichier (SmolDocling local)."""
    logger.info(f"--- Entrée /api/generate ({request.method}) ---")
    if request.method == 'OPTIONS':
        # Gérer la requête preflight CORS
        resp = make_response(jsonify(success=True), 200)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        logger.debug("/generate: Réponse OPTIONS envoyée.")
        return resp

    if not INITIALIZATION_SUCCESSFUL:
        logger.error("Requête /generate reçue mais serveur non initialisé.")
        return make_error_response("Serveur non prêt (échec initialisation)", 503)

    start_time = time.time()
    temp_file_path: Optional[str] = None # Pour le fichier uploadé

    try:
        if not request.is_json:
            logger.error("/generate: Requête invalide - Content-Type non JSON.")
            return make_error_response("Requête invalide: Content-Type doit être application/json.", 415)
        data = request.get_json()
        if not isinstance(data, dict):
             logger.error("/generate: Requête invalide - JSON n'est pas un objet.")
             return make_error_response("Requête invalide: JSON doit être un objet.", 400)

        conversation_history: List[Dict[str, Any]] = data.get('conversations', [])
        file_data_b64: Optional[str] = data.get('file_data')
        file_name: str = data.get('file_name', 'uploaded_file')
        options: Dict[str, bool] = data.get('options', {})

        # --- Validation et extraction du prompt utilisateur ---
        if not conversation_history or not isinstance(conversation_history, list):
             logger.error("/generate: 'conversations' requis.")
             return make_error_response("'conversations' requis (liste non vide).", 400)
        last_message = conversation_history[-1]
        # Gérer le cas où content peut être None ou pas une liste
        last_content = last_message.get("content")
        if last_message.get("role") != "user" or not isinstance(last_content, list) or not last_content:
             logger.error("/generate: Format dernier message invalide.")
             return make_error_response("Dernier message 'conversations' doit être 'user' avec 'content' (liste) non vide.", 400)

        user_prompt = ""
        for content_item in last_content:
             if content_item.get("type") == "text":
                  user_prompt = content_item.get("text", "").strip()
                  break
        if not user_prompt:
             logger.error("/generate: Aucun contenu textuel trouvé dans le dernier message.")
             return make_error_response("Aucun contenu textuel trouvé dans le dernier message utilisateur.", 400)

        logger.info(f"/generate: Prompt reçu='{user_prompt[:100]}...', Opts={options}, File={bool(file_data_b64)}")

        # --- Traitement du document uploadé (utilise SmolDocling localement) ---
        document_context = ""
        document_processing_error = None
        if file_data_b64 and options.get("document", True):
            logger.info(f"/generate: Traitement du fichier '{file_name}' activé (via SmolDocling local).")
            temp_file_handler = None # Pour gérer la fermeture correcte
            try:
                document_bytes = base64.b64decode(file_data_b64)
                file_suffix = Path(file_name).suffix if Path(file_name).suffix else '.tmp'
                # Créer et écrire dans le fichier temporaire
                temp_file_handler = tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, dir=CONFIG.TEMP_FOLDER)
                temp_file_path = temp_file_handler.name
                temp_file_handler.write(document_bytes)
                temp_file_handler.close() # Important de fermer avant de lire avec une autre fonction
                logger.debug(f"/generate: Fichier temporaire créé: {temp_file_path}")

                # Appelle la fonction qui utilise SmolDocling LOCALEMENT
                extracted_doc_text = extract_text_from_file(temp_file_path)

                if extracted_doc_text.startswith(("[Erreur", "[VLM SmolDocling:")):
                    document_processing_error = extracted_doc_text
                    logger.warning(f"/generate: Erreur/contenu vide extraction document (SmolDocling local): {document_processing_error}")
                elif not extracted_doc_text.strip():
                     document_processing_error = "[Document: Contenu vide extrait par SmolDocling]"
                     logger.warning(f"/generate: {document_processing_error}")
                else:
                    document_context = extracted_doc_text[:CONFIG.CONTEXT_MAX_CHARS_DOCUMENT]
                    logger.info(f"/generate: Document traité (SmolDocling local): {len(document_context)} caractères extraits.")
                    logger.debug(f"/generate: Extrait document (début): {document_context[:200]}...")

            except (base64.binascii.Error, FileProcessingError, VLMProcessingError) as file_err:
                logger.error(f"/generate: Erreur traitement fichier uploadé: {file_err}", exc_info=True)
                document_processing_error = f"[Erreur traitement fichier: {type(file_err).__name__}]"
            except Exception as e:
                 logger.error(f"/generate: Erreur inattendue traitement fichier: {e}", exc_info=True)
                 document_processing_error = f"[Erreur interne fichier: {type(e).__name__}]"
            finally: # Nettoyage fichier temporaire
                # Assurer la suppression même si temp_file_handler n'a pas été assigné correctement
                path_to_remove = temp_file_path if temp_file_path else (temp_file_handler.name if temp_file_handler else None)
                if path_to_remove and Path(path_to_remove).exists():
                    try:
                        os.remove(path_to_remove)
                        logger.debug(f"/generate: Fichier temporaire supprimé: {path_to_remove}")
                    except OSError as del_err:
                        logger.warning(f"/generate: Échec suppression fichier temp {path_to_remove}: {del_err}")
                # Réinitialiser au cas où
                temp_file_path = None
                temp_file_handler = None
        else:
             logger.info(f"/generate: Aucun fichier fourni ou option 'document' désactivée.")

        # --- Récupération contextes externes (RAG, Wiki - inchangé) ---
        rag_doc_context = ""; wiki_context = ""
        if options.get("rag", False):
            try: rag_doc_context = get_relevant_context_chromadb_docs(user_prompt)
            except RAGError as e: logger.error(f"/generate: Erreur RAG Documents: {e}")
            except Exception as e: logger.error(f"/generate: Erreur inattendue RAG Documents: {e}", exc_info=True)
        if options.get("wikipedia", False):
             try: wiki_context = get_wikipedia_context(user_prompt, max_chars=CONFIG.CONTEXT_MAX_CHARS_WIKI)
             except Exception as e: logger.error(f"/generate: Erreur Wikipedia: {e}", exc_info=True)

        # --- Combinaison du contexte EXTERNE ---
        context_parts = []
        if document_context: context_parts.append(f"Extrait du Document Fourni:\n{document_context}")
        if rag_doc_context: context_parts.append(f"Informations Pertinentes (Base Documentaire):\n{rag_doc_context}")
        if wiki_context: context_parts.append(f"Informations Wikipedia:\n{wiki_context}")
        combined_context_external = "\n\n---\n\n".join(context_parts).strip() if context_parts else None
        logger.debug(f"/generate: Contexte externe combiné (longueur={len(combined_context_external or '')}) avant appel LLM.")

        # --- Génération Texte via Serveur Externe (vlmserver @ 5001) ---
        try:
            logger.info(f"/generate: Appel generate_llm_response (vers serveur Texte externe)...")
            llm_response_text = generate_llm_response(
                prompt_user=user_prompt,
                conversation_history=conversation_history, # Passé mais non utilisé pour formatage prompt ici
                context_external=combined_context_external
            )
            processing_time = time.time() - start_time
            logger.info(f"Traitement /generate réussi en {processing_time:.2f} secondes.")
            # Retourne JSON avec clé "response" (ou "error")
            logger.info(f"--- Sortie /api/generate (Succès) ---")
            return jsonify({"response": llm_response_text})
        except LLMError as llm_err:
            logger.error(f"Échec génération Texte externe (/generate): {llm_err}")
            logger.info(f"--- Sortie /api/generate (Erreur LLM) ---")
            return make_error_response(f"Génération réponse: {llm_err}", 500)
        except Exception as llm_gen_err:
             logger.error(f"Erreur inattendue pendant appel Texte externe (/generate): {llm_gen_err}", exc_info=True)
             logger.info(f"--- Sortie /api/generate (Erreur Interne Appel) ---")
             return make_error_response("Erreur interne serveur lors de la génération.", 500)

    except Exception as e:
        logger.exception("Erreur inattendue majeure dans la route /api/generate:")
        logger.info(f"--- Sortie /api/generate (Erreur Serveur) ---")
        return make_error_response(f"Erreur interne serveur: {type(e).__name__}", 500)


# --- Route /api/sms_query (utilise generate_llm_response -> serveur externe) ---
@app.route('/api/sms_query', methods=['POST'])
def handle_sms_query_formdata():
    """
    Endpoint SMS. Utilise serveur externe pour la génération de réponse.
    Gère historique via EMBEDDINGS local et intègre Wikipedia.
    """
    logger.info(f"--- Entrée /api/sms_query ({request.method}) ---")
    if not INITIALIZATION_SUCCESSFUL:
        return make_error_response("Serveur non prêt", 503)

    sender = request.form.get('sender')
    message = request.form.get('message')
    if not sender or not message:
        return make_error_response("Données 'sender' ou 'message' manquantes", 400)
    logger.info(f"/sms_query: Reçu de {sender}: '{message[:100]}...'")

    # 1. Ajouter message USER à ChromaDB
    add_sms_to_history(sender, "user", message)

    # 2. Récupérer historique SMS pertinent (RAG)
    relevant_history_list = get_relevant_sms_history(
        sender=sender, query_message=message, n_results=CONFIG.SMS_HISTORY_RAG_N_RESULTS
    )
    history_context_str = format_history_context(
        relevant_history_list, CONFIG.CONTEXT_MAX_CHARS_SMS_HISTORY
    )

    # 3. Récupérer Contexte Wikipedia
    wiki_context_str = ""
    try:
        wiki_raw = get_wikipedia_context(message)
        if wiki_raw:
            wiki_context_str = f"Contexte Wikipedia pertinent:\n{wiki_raw[:CONFIG.CONTEXT_MAX_CHARS_WIKI]}\n---"
    except Exception as e: logger.error(f"Erreur recherche Wikipedia pour SMS {sender}: {e}", exc_info=True)

    # 4. Combiner contextes externes (Historique RAG + Wiki)
    combined_context_external = ""
    if history_context_str: combined_context_external += history_context_str + "\n\n"
    if wiki_context_str: combined_context_external += wiki_context_str
    combined_context_external = combined_context_external.strip() if combined_context_external else None
    logger.debug(f"/sms_query: Contexte externe combiné (longueur={len(combined_context_external or '')}).")

    # 5. Appel LLM via la fonction centralisée (vers serveur externe)
    llm_response_text = "[Erreur: Génération réponse échouée]"
    try:
        logger.info(f"/sms_query: Appel generate_llm_response (vers serveur Texte externe)...")
        llm_response_text = generate_llm_response(
            prompt_user=message,
            conversation_history=[], # Historique spécifique SMS est déjà dans le contexte RAG
            context_external=combined_context_external
        )

        # 6. Ajouter la réponse ASSISTANT à ChromaDB
        if not llm_response_text.startswith(("[Erreur:", "[Serveur VLM:")): # Check erreurs serveur externe aussi
             add_sms_to_history(sender, "assistant", llm_response_text)
        else: logger.warning(f"/sms_query: Réponse d'erreur non ajoutée à l'historique.")

        logger.info(f"Réponse serveur externe pour SMS {sender}: '{llm_response_text[:100]}...'")
        logger.info(f"--- Sortie /api/sms_query (Succès) ---")
        return jsonify({"reply": llm_response_text}) # Clé "reply" pour SMS

    except LLMError as llm_err: # Erreurs d'appel serveur externe
        logger.error(f"Échec génération Texte externe (/sms_query) pour {sender}: {llm_err}")
        logger.info(f"--- Sortie /api/sms_query (Erreur LLM) ---")
        return make_error_response(f"Génération réponse: {llm_err}", 500) # Clé "reply"
    except Exception as gen_err:
         logger.error(f"Erreur inattendue pendant appel Texte externe (/sms_query) pour {sender}: {gen_err}", exc_info=True)
         logger.info(f"--- Sortie /api/sms_query (Erreur Interne Appel) ---")
         return make_error_response("Interne serveur lors de la génération.", 500) # Clé "reply"

# --- Démarrage Serveur ---
if __name__ == '__main__':
    if initialize_models():
        logger.warning("="*60)
        logger.warning("Serveur Flask Principal (Mode Hybride) prêt.")
        logger.warning(f"Génération TEXTE: via serveur externe à {CONFIG.TEXT_GENERATION_SERVER_ENDPOINT}")
        logger.warning(f"Extraction IMAGE/PDF: via SmolDocling LOCAL ({'Activé' if MLX_VLM_AVAILABLE else 'Désactivé'})")
        logger.warning(f"MODELES LOCAUX: Embedding ({CONFIG.EMBEDDING_MODEL_NAME}), Reranker ({CONFIG.RERANKER_MODEL_NAME}), SmolDocling ({CONFIG.SMOL_VLM_MODEL_PATH if MLX_VLM_AVAILABLE else 'N/A'})")
        logger.warning(f"Traitement PDF local (conversion image): {'Activé (si Poppler OK)' if PDF2IMAGE_AVAILABLE else 'Désactivé'}")
        logger.warning(f"Collections ChromaDB: RAG='{CONFIG.CHROMA_COLLECTION_NAME}', SMS='{CONFIG.CHROMA_SMS_HISTORY_COLLECTION_NAME}'")
        logger.warning("ATTENTION: Assurez-vous que le serveur Texte externe (vlmserver.py @ 5001) est lancé séparément !")
        logger.warning("ATTENTION: Serveur Flask de développement.")
        logger.warning("="*60)
        # Port 5000 par défaut pour cette application principale
        app.run(host='0.0.0.0', port=500, debug=False, threaded=True)
    else:
        logger.critical("Démarrage Flask Principal annulé: Échec initialisation composants locaux critiques.")
        sys.exit(1)