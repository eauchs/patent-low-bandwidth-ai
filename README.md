# 🚀 HYBRID-ADVANCED

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ## 🌟 Présentation

HYBRID-ADVANCED est un projet combinant un serveur Flask pour un modèle de langage visuel (VLM) et une application Flask principale intégrant diverses fonctionnalités d'IA, telles que l'extraction de texte à partir d'images et de PDFs via un VLM local (SmolDocling), la recherche de documents pertinents (RAG) avec ChromaDB, la récupération d'informations de Wikipédia et la gestion de l'historique des conversations, et la possibilité de converser par SMS avec le backend. La génération de texte finale est gérée par un serveur VLM dédié.

Ce projet offre une architecture modulaire pour interagir avec des modèles de langage visuel, permettant des applications allant de la question-réponse basée sur des images et des documents à des systèmes de dialogue plus complexes.

## 🛠️ Fonctionnalités Clés

* **Serveur de Modèle de Langage Visuel (VLM) :**
    * Chargement à la demande de modèles VLM (actuellement configuré pour Mistral Small).
    * Génération de texte conditionnée par un prompt et une image (optionnelle).
    * API simple basée sur Flask pour interagir avec le modèle.
* **Application Principale :**
    * **Extraction de Texte Avancée :** Utilisation locale de SmolDocling pour extraire le texte d'images et de fichiers PDF.
    * **Récupération Augmentée par la Génération (RAG) :** Recherche de documents pertinents dans une base de connaissances ChromaDB.
    * **Intégration Wikipédia :** Récupération de contexte pertinent à partir de Wikipédia.
    * **Gestion de l'Historique :** Suivi de l'historique des conversations via ChromaDB.
    * **Génération de Texte Externe :** Utilisation d'un serveur Flask dédié pour la génération de texte finale, offrant une séparation des responsabilités et une flexibilité potentielle pour différents modèles de langage.
    * **API RESTful :** Exposition des fonctionnalités via des API Flask claires et documentées.

## 🚀 Démarrage

Suivez ces étapes pour configurer et exécuter le projet sur votre machine.

### Prérequis

Assurez-vous d'avoir les logiciels suivants installés :

* [Python 3.x](https://www.python.org/downloads/)
* [pip](https://pip.pypa.io/en/stable/installing/) (installé avec Python)
* [Git](https://git-scm.com/downloads)

### Installation

1.  **Cloner le Répertoire :**

    ```bash
    git clone https://github.com/eauchs/hybrid-advanced.git
    cd hybrid-advanced
    ```

2.  **Créer et Activer un Environnement Virtuel (Recommandé) :**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Sur Linux/macOS
    # venv\Scripts\activate  # Sur Windows
    ```

3.  **Installer les Dépendances :**

    ```bash
    # Pour le serveur VLM (exécuter dans le répertoire contenant vlmserver.py)
    cd server
    pip install -r requirements.txt
    cd ..

    # Pour l'application principale (exécuter dans le répertoire racine du projet)
    cd hybrid-advanced
    pip install -r requirements.txt
    cd ..
    ```

4.  **Configuration :**

    * Créez un fichier `.env` dans le répertoire racine de chaque application (serveur VLM et application principale) et configurez les variables d'environnement nécessaires. Voici un exemple de variables que vous pourriez avoir besoin de configurer :

        **Pour le serveur VLM (`server/.env`) :**

        ```env
        VLM_MODEL_PATH="mlx-community/Mistral-Small-3.1-24B-Instruct-2503-3bit"
        SERVER_HOST="0.0.0.0"
        SERVER_PORT=5001
        ```

        **Pour l'application principale (`main_app_hybrid/.env`) :**

        ```env
        TEMP_FOLDER="/tmp"
        CHROMA_DB_PATH="chromadb_base"
        CHROMA_COLLECTION_NAME="my_collection"
        CHROMA_SMS_HISTORY_COLLECTION_NAME="sms_history_collection"
        EMBEDDING_MODEL_NAME="ibm-granite/granite-embedding-278m-multilingual"
        RERANKER_MODEL_NAME="ibm-research/re2g-reranker-nq"
        SMOL_VLM_MODEL_PATH="ds4sd/SmolDocling-256M-preview-mlx-bf16" # Si vous utilisez SmolDocling localement
        TEXT_GENERATION_SERVER_ENDPOINT="http://localhost:5001/api/vlm_generate"
        REQUEST_TIMEOUT=1000
        SMOL_VLM_MAX_TOKENS=1024
        SMOL_VLM_TEMPERATURE=0.0
        WIKIPEDIA_LANG="fr"
        WIKIPEDIA_MAX_CHARS=8000
        RAG_N_RESULTS=15
        RAG_MIN_TOKEN_COUNT=10
        RAG_TOP_K_RERANKED=10
        CONTEXT_MAX_CHARS_DOCUMENT=6000
        CONTEXT_MAX_CHARS_RAG=10000
        CONTEXT_MAX_CHARS_WIKI=4000
        SMS_HISTORY_RAG_N_RESULTS=10
        CONTEXT_MAX_CHARS_SMS_HISTORY=5000
        PDF_MAX_PAGES=10
        PDF_DPI=200
        PDF_CONVERT_TIMEOUT=120
        ```

        **Note :** Ajustez les chemins et les valeurs en fonction de votre configuration.

### Exécution

1.  **Démarrer le Serveur VLM :**

    ```bash
    cd server
    python server.py
    ```

    Le serveur devrait démarrer sur `http://0.0.0.0:5001` (ou l'adresse et le port configurés).

2.  **Démarrer l'Application Principale :**

    ```bash
    cd main_app_hybrid
    python main_app_hybrid.py
    ```

    L'application principale devrait démarrer sur `http://0.0.0.0:500` (par défaut).

## ⚙️ Utilisation

Décrivez ici comment interagir avec votre application. Par exemple :

* Pour interagir avec l'application principale, vous pouvez envoyer des requêtes POST à l'endpoint `/api/generate` avec un payload JSON contenant l'historique des conversations, les données de fichier (optionnel) et les options de traitement (RAG, Wikipédia, document).
* Pour le serveur VLM, vous pouvez envoyer des requêtes POST à `/api/vlm_generate` avec un prompt et une image base64 (optionnelle) pour obtenir une réponse textuelle.

**Exemple de requête à l'application principale (`/api/generate`) :**

```json
{
  "conversations": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Quel est le contenu de cette image ?" }
      ]
    }
  ],
  "file_data": "...", // Votre image en base64
  "file_name": "image.png",
  "options": {
    "document": true,
    "rag": false,
    "wikipedia": false
  }
}

{
  "prompt": "Décrivez cette image.",
  "image_base64": "...", // Votre image en base64 (optionnel)
  "max_tokens": 256,
  "temperature": 0.7
}

# hybrid-advanced
