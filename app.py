import streamlit as st
import streamlit.components.v1 as components
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from sacrebleu import sentence_bleu

st.set_page_config(
    page_title="Générateur de Poésie IA",
    page_icon="📝",
    layout="wide"
)

##############################
# 1. Chargement des métriques
##############################
rouge_metric = evaluate.load("rouge")

def compute_rouge_score(text_ref, text_hyp):
    """
    Calcule les scores ROUGE entre un texte de référence et un texte hypothèse.
    """
    results = rouge_metric.compute(
        predictions=[text_hyp], 
        references=[text_ref]
    )
    return results

def compute_bleu_score(text_ref, text_hyp):
    """
    Calcule un score BLEU entre un texte de référence et un texte hypothèse.
    """
    reference = text_ref.strip()
    hypothesis = text_hyp.strip()
    bleu = sentence_bleu(hypothesis, [reference])
    return bleu.score / 100.0

##############################
# 2. Chargement des modèles
##############################
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

MODEL_GPT2 = "gpt2"
MODEL_GPTNEO = "EleutherAI/gpt-neo-125M"

tokenizer_gpt2, model_gpt2 = load_model(MODEL_GPT2)
tokenizer_gptneo, model_gptneo = load_model(MODEL_GPTNEO)

##############################
# 3. Fonction de génération
##############################
def generate_text(tokenizer, model, prompt, max_length=50, temperature=1.0, top_p=0.9):
    """
    Génère un texte à partir d'un prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    model.eval()
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

##############################
# 4. Comparaison BLEU/ROUGE
##############################
def comparer_textes(text_gpt2, text_gptneo):
    """
    Compare les textes avec BLEU et ROUGE.
    """
    bleu = compute_bleu_score(text_gpt2, text_gptneo)
    rouge = compute_rouge_score(text_gpt2, text_gptneo)
    
    return {
        "BLEU": bleu,
        "ROUGE": rouge
    }

##############################
# 5. Interface Streamlit améliorée
##############################
# Configuration de la page - DOIT ÊTRE AU DÉBUT

# Session state pour le thème
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def get_theme_colors():
    """Retourne les couleurs selon le thème"""
    if st.session_state.theme == 'dark':
        return {
            'bg_color': '#1e1e1e',
            'text_color': '#ffffff',
            'card_bg': '#2d2d2d',
            'card_border': '#3d3d3d',
            'poem_bg': '#363636'
        }
    return {
        'bg_color': '#f1f5f9',
        'text_color': '#1e293b',
        'card_bg': 'white',
        'card_border': '#e2e8f0',
        'poem_bg': '#f8fafc'
    }

def create_poem_display(title, content, colors):
    """
    Crée une carte HTML pour afficher un poème avec effet de typing
    """
    unique_id = f"poem-{title.lower().replace(' ', '-')}"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .poem-card {{
                background-color: {colors['card_bg']};
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: 1px solid {colors['card_border']};
            }}
            .poem-title {{
                color: {colors['text_color']};
                font-size: 1.5rem;
                margin-bottom: 15px;
                font-weight: 600;
                border-bottom: 2px solid {colors['card_border']};
                padding-bottom: 10px;
                font-family: Georgia, serif;
            }}
            .poem-content {{
                font-family: Georgia, serif;
                white-space: pre-wrap;
                line-height: 1.6;
                color: {colors['text_color']};
                font-size: 1.1rem;
                background-color: {colors['poem_bg']};
                padding: 15px;
                border-radius: 8px;
                min-height: 200px;
            }}
        </style>
    </head>
    <body>
        <div class="poem-card">
            <h3 class="poem-title">{title}</h3>
            <div id="{unique_id}" class="poem-content"></div>
        </div>

        <script>
            function sleep(ms) {{
                return new Promise(resolve => setTimeout(resolve, ms));
            }}
            
            async function typeText(text, elementId, speed = 50) {{
                const element = document.getElementById(elementId);
                const words = text.split(' ');
                element.innerHTML = '';
                
                for (let i = 0; i < words.length; i++) {{
                    element.innerHTML += words[i];
                    if (i < words.length - 1) {{
                        element.innerHTML += ' ';
                    }}
                    await sleep(speed);
                }}
            }}

            document.addEventListener('DOMContentLoaded', function() {{
                typeText(`{content.replace('"', '\\"')}`, '{unique_id}');
            }});
        </script>
    </body>
    </html>
    """
    
    return components.html(html, height=400, scrolling=True)


def create_metric_card(title, value, description, colors):
    """Crée une carte pour afficher une métrique"""
    # Calculer la couleur en fonction de la valeur
    color = "#22c55e" if value >= 0.4 else "#f97316" if value >= 0.2 else "#ef4444"
    
    return f"""
    <div style="
        background-color: {colors['card_bg']};
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid {colors['card_border']};
    ">
        <h4 style="
            color: {colors['text_color']};
            font-size: 1.1rem;
            margin-bottom: 10px;
            font-weight: 600;
        ">{title}</h4>
        <div style="
            font-size: 1.8rem;
            font-weight: bold;
            color: {color};
            margin: 10px 0;
        ">{value:.3f}</div>
        <div style="
            font-size: 0.9rem;
            color: {colors['text_color']};
            opacity: 0.8;
        ">{description}</div>
    </div>
    """

# [Garder les fonctions existantes pour le modèle et les métriques]

def main():
    colors = get_theme_colors()
    
    # Style CSS personnalisé
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {colors['bg_color']};
            color: {colors['text_color']};
        }}
        .main {{
            padding: 2rem;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # En-tête avec bouton de thème
    col_title, col_theme = st.columns([4, 1])
    with col_title:
        st.title("🎨 Générateur de Poésie avec IA")
    with col_theme:
        if st.button("🌓 Changer de thème"):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
            st.experimental_rerun()

    st.markdown(f"""
    <div style='background-color: {colors['card_bg']}; color: {colors['text_color']}; 
                padding: 20px; border-radius: 10px; margin-bottom: 20px; 
                border: 1px solid {colors['card_border']};'>
        Entrez un <strong>mot-clé</strong> ou une <strong>phrase</strong>, puis ajustez les paramètres pour générer deux poèmes uniques.
    </div>
    """, unsafe_allow_html=True)

    user_prompt = st.text_input("✨ Votre inspiration", 
                               value="Write a romantic free verse poem about love and freedom",
                               help="Entrez un thème, un mot-clé ou une phrase pour inspirer les poèmes")

    # Paramètres dans un container élégant
    with st.container():
        st.markdown("### ⚙️ Paramètres de génération")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_length = st.slider("📏 Longueur maximale", 20, 200, 50, 5,
                                 help="Nombre maximum de tokens générés")
        with col2:
            temperature = st.slider("🌡️ Température", 0.7, 1.5, 1.0, 0.1,
                                  help="Contrôle la créativité (plus élevé = plus créatif)")
        with col3:
            top_p = st.slider("🎯 Top-p", 0.5, 1.0, 0.9, 0.05,
                            help="Contrôle la diversité des mots choisis")

    if st.button("🚀 Générer les poèmes", use_container_width=True):
        st.session_state.generated = True
        with st.spinner("✨ Création en cours..."):
            poem_gpt2 = generate_text(tokenizer_gpt2, model_gpt2, user_prompt, 
                                    max_length=max_length, 
                                    temperature=temperature, 
                                    top_p=top_p)
            poem_gptneo = generate_text(tokenizer_gptneo, model_gptneo, user_prompt, 
                                      max_length=max_length, 
                                      temperature=temperature, 
                                      top_p=top_p)
            
            # Stockage des poèmes dans session_state
            st.session_state.poem_gpt2 = poem_gpt2
            st.session_state.poem_gptneo = poem_gptneo

    # Affichage des poèmes uniquement s'ils ont été générés
    if hasattr(st.session_state, 'generated') and st.session_state.generated:
        col1, col2 = st.columns(2)
        
        with col1:
            create_poem_display("🤖 Poème GPT-2", st.session_state.poem_gpt2, colors)
        
        with col2:
            create_poem_display("🤖 Poème GPT-Neo", st.session_state.poem_gptneo, colors)

        # Calcul et affichage des scores
        scores = comparer_textes(st.session_state.poem_gpt2, st.session_state.poem_gptneo)
        
        st.markdown(f"### 📊 Analyse de similarité")
        
        # BLEU score
        st.markdown(create_metric_card(
            "Score BLEU",
            scores["BLEU"],
            "Mesure la similarité des séquences de mots",
            colors
        ), unsafe_allow_html=True)

        # ROUGE scores
        col_rouge1, col_rouge2 = st.columns(2)
        with col_rouge1:
            for metric in ['rouge1', 'rouge2']:
                st.markdown(create_metric_card(
                    f"Score {metric.upper()}",
                    scores["ROUGE"][metric],
                    "Chevauchement des unigrammes/bigrammes",
                    colors
                ), unsafe_allow_html=True)

        with col_rouge2:
            for metric in ['rougeL', 'rougeLsum']:
                st.markdown(create_metric_card(
                    f"Score {metric.upper()}",
                    scores["ROUGE"][metric],
                    "Plus longue sous-séquence commune",
                    colors
                ), unsafe_allow_html=True)

if __name__ == "__main__":
    main()