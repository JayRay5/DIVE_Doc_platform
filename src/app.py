import gradio as gr
import requests
import io
import os

#API_URL = "http://127.0.0.1:8000/ask"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

# --- COULEURS ICCV & THEME PERSONNALISÉ ---
# Une palette professionnelle : Bleu profond, Teal moderne, et Gris Ardoise
iccv_blue_primary = "#005b96"   # Bleu institutionnel fort
iccv_blue_secondary = "#0088cc" # Bleu plus clair pour les dégradés
iccv_bg_gray = "#f8fafc"        # Fond très clair, légèrement bleuté

# Définition du thème Gradio
# On part du thème Soft et on injecte nos couleurs
theme = gr.themes.Soft(
    # On force notre bleu comme couleur primaire
    primary_hue=gr.themes.Color(
        c50= "#e6f0f7", c100="#b3cde0", c200="#99bdd6", c300="#80adcc", 
        c400="#669cc2", c500=iccv_blue_primary, c600="#00528a", c700="#004a7a", 
        c800="#003d66", c900="#002e4d", c950="#001f33"
    ),
    neutral_hue="slate", # Slate donne un ton gris-bleu professionnel aux textes
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
).set(
    # Ajustements fins pour un look "Premium"
    body_background_fill=iccv_bg_gray,
    block_background_fill="#ffffff",
    block_border_width="0px",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
    button_large_radius="8px"
)

# --- CUSTOM CSS POUR LA BANNIÈRE ICCV ---
custom_css = f"""
#title-container {{
    text-align: center;
    padding: 40px 20px;
    /* Dégradé professionnel ICCV */
    background: linear-gradient(135deg, {iccv_blue_primary} 0%, {iccv_blue_secondary} 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 10px 25px -5px rgba(0, 91, 150, 0.3);
}}
#logo-text {{
    font-size: 3.5em;
    font-weight: 800;
    letter-spacing: -1px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}}
#subtitle-text {{
    font-size: 1.3em;
    font-weight: 400;
    opacity: 0.9;
    margin-top: 10px;
}}
/* Petit badge style conférence */
.badge-iccv {{
    display: inline-block;
    background-color: rgba(255,255,255,0.15);
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9em;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.3);
}}
"""

def app():
    def answer_question(image, question):
        if image is None:
            return "⚠️ Error : Please upload an image before to submit."
    
        if not question:
            return "⚠️ Error : Please write a question before to submit."
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        payload = {"question": question}
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

        try:
            print(f"[INFO] Sending the request to {API_URL}...")
            response = requests.post(API_URL, data=payload, files=files,timeout=480)
            if response.status_code == 200:
                result = response.json()
                return result.get("answer", "Pas de réponse trouvée.")
            else:
                return f"Server error : ({response.status_code}) : {response.text}"
            
        except requests.exceptions.Timeout:
            return "Timeout: The model took too long to respond."
        except requests.exceptions.ConnectionError:
            return "API Connection issue."
        except Exception as e:
            return f"Unexpected Error : {str(e)}"

    with gr.Blocks(theme=theme, css=custom_css, title="DIVE-Doc ICCV'25 Demo") as demo:
            
            # 1. Header Bannière "ICCV Style"
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(
                        f"""
                        <div id="title-container">
                            <div class="badge-iccv">🎉 Presented at ICCV 2025</div>
                            <div id="logo-text">🤖 DIVE-Doc</div>
                            <div id="subtitle-text">End-to-End OCR-Free Document Visual Question Answering</div>
                        </div>
                        """
                    )

            # 2. Zone Principale (2 Colonnes avec des titres propres)
            with gr.Row():
                # Colonne GAUCHE
                with gr.Column(scale=5, variant="panel"):
                    gr.Markdown("### 📄 1. Upload Document")
                    input_img = gr.Image(
                        type="pil", 
                        label="", # Label caché car le titre Markdown suffit
                        height=450,
                        sources=["upload", "clipboard"])

                # Colonne DROITE
                with gr.Column(scale=4, variant="panel"):
                    gr.Markdown("### 💬 2. Ask & Analyze")
                    input_question = gr.Textbox(
                        label="Question", 
                        placeholder="e.g., What is the invoice total amount?",
                        lines=3
                    )
                    
                    submit_btn = gr.Button("🚀 Run Inference", variant="primary", size="lg")
                    
                    #gr.Separator() # Une ligne de séparation propre
                    
                    gr.Markdown("### 💡 Model Prediction")
                    output_answer = gr.Textbox(
                        label="",
                        show_label=False,
                        lines=4,
                        interactive=False,
                        placeholder="The answer will appear here..."
                    )

        

            # 4. Exemples
            gr.Markdown("### 🧪 Test with Official Examples")
            gr.Examples(
                # REMPLACE PAR TES VRAIS FICHIERS
                examples=[
                # ["examples/figure1_iccv.png", "What is the main trend in chart A?"],
                # ["examples/table2_results.png", "What is the accuracy for DIVE-Doc on DocVQA?"],
                ],
                inputs=[input_img, input_question],
                label="Click to load example",
                cache_examples=False # Important si pas de GPU sur la machine Gradio
            )

            # --- LOGIQUE ---
            submit_btn.click(
                fn=answer_question, 
                inputs=[input_img, input_question], 
                outputs=[output_answer]
            )
            
            input_question.submit(
                fn=answer_question, 
                inputs=[input_img, input_question], 
                outputs=[output_answer]
            )

        # Lancement
    print(f"Starting ICCV'25 Themed Gradio on port 7860...")
    demo.launch(server_name="0.0.0.0", server_port=7860, favicon_path=None) # Tu peux ajouter un favicon.ico si tu en as un

if __name__ == "__main__":
    app()