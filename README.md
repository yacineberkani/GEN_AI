# Projet : Génération de poèmes avec GPT-2 et GPT-Neo

Ce projet a pour objectif de générer des poèmes (ou tout autre texte créatif) à partir d’un mot-clé ou d’une phrase fournie par l’utilisateur. Deux modèles de langage, **GPT-2** et **GPT-Neo**, sont comparés afin d’évaluer leurs similarités via les scores **BLEU** et **ROUGE**. L’interface graphique est développée avec **Streamlit**.

---

## Fonctionnalités

1. **Interface utilisateur** simple (Streamlit) :
   - Zone de texte pour saisir un mot-clé ou une phrase (prompt).
   - Bouton pour générer le poème.
   - Affichage de deux poèmes (GPT-2 et GPT-Neo).
   - Visualisation des scores **BLEU** et **ROUGE** sous forme de graphiques (Altair).

2. **Génération de texte** :
   - Utilise des modèles pré-entraînés disponibles via la bibliothèque `transformers` (Hugging Face).
   - Paramètres ajustables (longueur de séquence, température, top-p) pour une génération plus ou moins créative.

3. **Évaluation de similarité** :
   - **BLEU** : calculé avec la bibliothèque `sacrebleu`.
   - **ROUGE** : calculé avec la bibliothèque `evaluate` (Hugging Face).

---

## Installation

1. **Cloner** le dépôt ou récupérer les fichiers du projet.
2. (Optionnel, mais recommandé) Créer et activer un **environnement virtuel** :
   ```bash
   python -m venv env
   # Windows
   .\env\Scripts\activate
   # OU (Mac/Linux)
   source env/bin/activate
3. Installer les dépendances :
   ````
   pip install -r requirements.txt
   ````

4. (Optionnel) Installer PyTorch adapté à votre GPU/CUDA (voir [la documentation PyTorch](https://pytorch.org/) pour plus d'informations).

## Utilisation
1. Lancement de l’application Streamlit :
````
   streamlit run app.py
   ````
- `app.py` est le fichier principal qui contient le code Streamlit et les fonctions de génération/évaluation.
2. Interface graphique :

- Ouvrez http://localhost:8501 dans votre navigateur.
- Entrez un mot-clé ou une phrase (ex. “Amour et liberté”).
- Sélectionnez les paramètres de génération (longueur, température, top-p).
- Cliquez sur Générer le poème.
- Les deux poèmes s’afficheront, suivis de leurs scores BLEU et ROUGE sous forme de diagrammes à barres (Altair).
  
## Arborescence (exemple)
   `````bash
   ├── app.py                  # Script principal Streamlit
   ├── requirements.txt        # Liste des dépendances (optionnel)
   ├── README.md               # Le présent fichier
   └── ...
``````


## Explications techniques

1. Chargement des modèles (**GPT-2** & **GPT-Neo**) :

 - Via `AutoTokenizer` et `AutoModelForCausalLM` (bibliothèque `transformers`).
 - Les poids sont téléchargés depuis Hugging Face, puis mis en cache localement.

2. Génération de texte :

- Fonction `generate_text(...)` utilisant `model.generate()` de **PyTorch/Transformers**.
- Paramètres tels que `max_length`, `temperature`, `top_p` pour contrôler la créativité.

3. Calcul des scores :

- **BLEU :** fonction `compute_bleu_score` (utilisant `sacrebleu.sentence_bleu`).
- **ROUGE :** fonction `compute_rouge_score` (utilisant la librairie `evaluate`).

4. Comparaison :

- La fonction `comparer_textes(...)` renvoie un dictionnaire contenant les scores BLEU et ROUGE sous différentes références (**GPT-2** comme ref, **GPT-Neo** comme ref).
  
5. Visualisation :

- **Streamlit** pour afficher les textes générés (scores `BLEU` et `ROUGE`) de manière interactive et claire.

--------------------------------------------------
## Licence

Ce projet est un projet académique réalisé dans le cadre d’une formation. Il n’est pas distribué sous une licence particulière.

---------------------------------------------------

**Merci d’avoir consulté ce README.**

**N’hésitez pas à contribuer en proposant des améliorations ou en signalant des bugs !**