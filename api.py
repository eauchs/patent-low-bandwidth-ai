import os
import base64
import tempfile
import logging
import sys
from io import BytesIO
from pathlib import Path
import traceback # Pour un meilleur logging d'erreur

from flask import Flask, request, jsonify
from PIL import Image

# Assurez-vous que mlx_vlm est installé et importable
try:
    from mlx_vlm.utils import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    MLX_VLM_AVAILABLE = True
except ImportError as e:
    logging.error(f"Erreur: Impossible d'importer mlx_vlm. Assurez-vous qu'il est installé. {e}")
    MLX_VLM_AVAILABLE = False
    # Définir des stubs
    load = lambda path, **kwargs: (None, None, None) # Modifié pour inclure config
    generate = lambda *args, **kwargs: "[Erreur: mlx_vlm non disponible]"
    apply_chat_template = lambda proc, conf, p, **kwargs: p
    load_config = lambda path, **kwargs: None

# --- Configuration ---
VLM_MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-3bit")
SERVER_HOST = os.environ.get("VLM_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("VLM_SERVER_PORT", 5001))

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Initialisation Flask ---
app = Flask(__name__)

# --- Chargement du Modèle VLM (au démarrage) ---
vlm_model = None
vlm_processor = None
vlm_config = None

def initialize_vlm():
    global vlm_model, vlm_processor, vlm_config
    if not MLX_VLM_AVAILABLE:
        logger.critical("mlx_vlm n'est pas disponible. Le serveur VLM ne peut pas démarrer.")
        return False

    logger.info(f"Chargement du modèle VLM : {VLM_MODEL_PATH}...")
    try:
        vlm_model, vlm_processor = load(VLM_MODEL_PATH, trust_remote_code=True)
        vlm_config = load_config(VLM_MODEL_PATH, trust_remote_code=True)
        if vlm_model is None or vlm_processor is None or vlm_config is None:
             raise ValueError("Le chargement du modèle, processeur ou config a retourné None.")
        logger.info("Modèle VLM chargé avec succès.")
        return True
    except Exception as e:
        logger.critical(f"Erreur critique lors du chargement du modèle VLM : {e}", exc_info=True)
        return False

# --- Route API (Modifiée) ---
@app.route('/api/vlm_generate', methods=['POST'])
def handle_vlm_generate():
    if vlm_model is None or vlm_processor is None or vlm_config is None:
         logger.error("Requête reçue mais le modèle VLM n'est pas chargé.")
         return jsonify({"error": "Serveur VLM non initialisé correctement."}), 503

    if not request.is_json:
        logger.warning("Requête reçue non JSON.")
        return jsonify({"error": "La requête doit être au format JSON"}), 400

    data = request.get_json()
    prompt_text = data.get('prompt')
    image_b64 = data.get('image_base64') # Image est maintenant optionnelle
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)

    # Le prompt est toujours requis
    if not prompt_text:
        logger.warning("Requête invalide: 'prompt' manquant.")
        return jsonify({"error": "Le champ 'prompt' est requis."}), 400

    image_path = None # Pour le cleanup
    has_image = bool(image_b64) # Vérifie si l'image a été fournie

    try:
        image_to_generate = [] # Liste vide par défaut (pas d'image)

        # 1. Traiter l'image SEULEMENT si elle est fournie
        if has_image:
            logger.info("Traitement de l'image fournie...")
            try:
                image_bytes = base64.b64decode(image_b64)
                img_pil = Image.open(BytesIO(image_bytes))
                # Utiliser le format détecté par PIL ou PNG par défaut
                suffix = f".{img_pil.format.lower()}" if img_pil.format else ".png"
            except (base64.binascii.Error, IOError, Exception) as decode_err:
                 logger.error(f"Erreur décodage/ouverture image base64: {decode_err}")
                 return jsonify({"error": "Impossible de décoder ou lire l'image base64."}), 400

            # Créer un fichier temporaire sécurisé
            try:
                fd, image_path = tempfile.mkstemp(suffix=suffix, prefix="vlm_img_")
                os.close(fd)
                img_pil.save(image_path)
                image_to_generate = [image_path] # La liste contient le chemin si image présente
                logger.info(f"Image temporaire créée : {image_path}")
            except Exception as save_err:
                 logger.error(f"Erreur sauvegarde image temporaire: {save_err}")
                 return jsonify({"error": "Erreur interne lors de la sauvegarde de l'image."}), 500
        else:
            logger.info("Aucune image fournie, génération texte seul.")


        # 2. Préparer le prompt avec le template (adapte num_images)
        num_images = 1 if has_image else 0
        formatted_prompt = apply_chat_template(
            vlm_processor, vlm_config, prompt_text, num_images=num_images
        )
        logger.debug(f"Prompt formaté (début): {formatted_prompt[:100]}...")

        # 3. Appeler la fonction de génération mlx_vlm
        logger.info(f"Lancement de la génération {'VLM' if has_image else 'texte'} (max_tokens={max_tokens}, temp={temperature})...")
        output = generate(
             vlm_model,
             vlm_processor,
             formatted_prompt,
             image=image_to_generate, # Liste vide si pas d'image
             temperature=temperature,
             max_tokens=max_tokens,
             verbose=False
        )
        logger.info(f"Génération {'VLM' if has_image else 'texte'} terminée avec succès.")

        # 4. Retourner la réponse
        return jsonify({"response": output})

    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}", exc_info=True)
        # Afficher la traceback complète dans les logs serveur pour le débogage
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Erreur interne du serveur: {str(e)}"}), 500

    finally:
        # 5. Nettoyer le fichier image temporaire s'il a été créé
        if image_path and Path(image_path).exists():
            try:
                os.remove(image_path)
                logger.info(f"Image temporaire supprimée: {image_path}")
            except OSError as del_err:
                logger.warning(f"Échec de la suppression de l'image temporaire {image_path}: {del_err}")


# --- Démarrage Serveur ---
if __name__ == '__main__':
    if initialize_vlm():
        logger.info(f"Serveur VLM (Mistral Small) démarré sur http://{SERVER_HOST}:{SERVER_PORT}")
        logger.info("Endpoint /api/vlm_generate accepte 'prompt' (requis) et 'image_base64' (optionnel).")
        app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
    else:
         logger.critical("Échec de l'initialisation du serveur VLM.")
         sys.exit(1)