import gradio as gr
import requests
import io
import os
from pathlib import Path

# --- Backend url ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

# META-DATA
PROJECT_INFO = {
    "title": "DIVE-Doc",
    "paper_url": "https://openaccess.thecvf.com/content/ICCV2025W/VisionDocs/html/Bencharef_DIVE-Doc_Downscaling_foundational_Image_Visual_Encoder_into_hierarchical_architecture_for_ICCVW_2025_paper.html",
    "repo_url": "https://github.com/JayRay5/DIVE-Doc",
    "weights_url": "https://huggingface.co/JayRay5/DIVE-Doc-FRD",
    "citation": """@InProceedings{Bencharef_2025_ICCV,
    author    = {Bencharef, Rayane and Rahiche, Abderrahmane and Cheriet, Mohamed},
    title     = {DIVE-Doc: Downscaling foundational Image Visual Encoder into hierarchical architecture for DocVQA},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {7547-7556}
}""",
}

# --- COLORS ---
COLORS = {
    "primary": "#005b96",
    "secondary": "#0088cc",
    "main": "#8db8ce",
    "inference-btn": "#5ea7cb",
    "bg": "#f8fafc",
    "purple": "#a42967",
    "gold": "#FFD700",
}

# --- CSS  ---
CUSTOM_CSS = f"""
/* Conteneur Principal En-tête */
#title-container {{
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    gap: 40px;
    padding: 30px 40px;
    background-color: {COLORS["main"]};
    color: white;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 91, 150, 0.3);
    flex-wrap: wrap; /* Permet le responsive */
}}

/* Zone Logo + Titres */
.header-left {{
    display: flex;
    align-items: center;
    gap: 30px;
}}

.logo-img {{
    height: 80px;
    width: auto;
    filter: drop-shadow(0 4px 6px rgba(0,0,0,0.2));
}}

.titles-wrapper {{
    display: flex;
    flex-direction: column;
}}

#project-title {{
    font-size: 3em;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -1px;
}}

#project-subtitle {{
    font-size: 1.1em;
    font-weight: 400;
    opacity: 0.95;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}}

/* Badge Award */
.award-badge {{
    background: rgba(255, 215, 0, 0.15);
    border: 1px solid {COLORS["gold"]};
    color: {COLORS["gold"]};
    padding: 4px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85em;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}}

/* Link Button */
.links-container {{
    display: flex;
    gap: 15px;
}}

.link-btn {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.1);
    padding: 10px 15px;
    border-radius: 8px;
    text-decoration: none;
    color: white;
    transition: all 0.2s ease;
    border: 1px solid rgba(255, 255, 255, 0.2);
    min-width: 70px;
}}

.link-btn:hover {{
    background: rgba(255, 255, 255, 0.25);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}

.link-btn img {{
    height: 24px;
    margin-bottom: 4px;
    filter: brightness(0) invert(1); /* Rend les icônes blanches */
}}
.link-btn-color img {{
     filter: none;
}}

.link-text {{
    font-size: 0.75em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
#inference_btn {{
    background-color: {COLORS["inference-btn"]} !important; /* Un peu plus foncé au survol */
}}
#inference_btn:hover {{
    background-color: {COLORS["purple"]} !important; /* Un peu plus foncé au survol */
}}

/* RESPONSIVE DESIGN (Mobile & Tablet) */
@media (max-width: 1200px) {{
    #title-container {{
        flex-direction: column;
        text-align: center;
        padding: 20px;
    }}
    .header-left {{
        flex-direction: column;
        gap: 15px;
    }}
    .titles-wrapper {{ align-items: center; }}
    #project-subtitle {{ flex-direction: column; }}
    .links-container {{ margin-top: 15px; }}
}}
"""


def answer_question(image, question):
    if image is None:
        return "⚠️ Error: Please upload an image first."
    if not question:
        return "⚠️ Error: Please enter a question."

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    payload = {"question": question}
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

    try:
        print(f"[INFO] Sending request to {API_URL}...")
        response = requests.post(API_URL, data=payload, files=files, timeout=480)

        if response.status_code == 200:
            result = response.json()
            return result.get("answer", "No answer found.")
        else:
            return f"Server Error ({response.status_code}): {response.text}"

    except requests.exceptions.Timeout:
        return "⏳ Timeout: The model is taking too long (try a smaller image)."
    except requests.exceptions.ConnectionError:
        return "⛔ Connection Error: Is the backend running?"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


def build_app():
    theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#e6f0f7",
            c100="#b3cde0",
            c200="#99bdd6",
            c300="#80adcc",
            c400="#669cc2",
            c500=COLORS["primary"],
            c600="#00528a",
            c700="#004a7a",
            c800="#003d66",
            c900="#002e4d",
            c950="#001f33",
        ),
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    ).set(
        body_background_fill=COLORS["bg"],
        button_primary_background_fill="*primary_500",
        button_primary_text_color="white",
        button_large_radius="8px",
    )

    with gr.Blocks(
        theme=theme, css=CUSTOM_CSS, title=f"{PROJECT_INFO['title']} Demo"
    ) as demo:
        # --- HEADER HTML ---
        gr.HTML(f"""
        <div id="title-container">
            <div class="header-left">
                <img src="/gradio_api/file=assets/iccv-navbar-logo.svg" class="logo-img" alt="ICCV Logo">
                <div class="titles-wrapper">
                    <div id="project-title">{PROJECT_INFO["title"]}</div>
                    <div id="project-subtitle">
                        <span>Spotlight at <b>VisionDocs Workshop</b></span>
                        <div class="award-badge">🏆 Best Paper Award</div>
                    </div>
                </div>
            </div>
            
            <div class="links-container">
                <a href="{PROJECT_INFO["paper_url"]}" target="_blank" class="link-btn">
                    <img src="/gradio_api/file=assets/cropped-cvf-s.png" alt="Paper">
                    <span class="link-text">Paper</span>
                </a>
                <a href="{PROJECT_INFO["repo_url"]}" target="_blank" class="link-btn">
                    <img src="/gradio_api/file=assets/github-mark.svg" alt="Code">
                    <span class="link-text">Code</span>
                </a>
                <a href="{PROJECT_INFO["weights_url"]}" target="_blank" class="link-btn link-btn-color">
                    <img src="/gradio_api/file=assets/hf-logo-pirate.png" alt="Weights">
                    <span class="link-text">Weights</span>
                </a>
            </div>
        </div>
        """)

        # --- Main Area ---
        with gr.Row():
            # --- Left Column ---
            with gr.Column(scale=5):
                gr.Markdown("### 📄 Upload Document")
                input_img = gr.Image(
                    type="pil",
                    label="Document",
                    height=500,
                    sources=["upload", "clipboard"],
                )

            # --- Right Column ---
            with gr.Column(scale=4):
                gr.Markdown("### 💬 Ask a Question")
                input_question = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What is the total amount?",
                    lines=3,
                )

                submit_btn = gr.Button(
                    "🚀 Run Inference",
                    variant="primary",
                    elem_id="inference_btn",
                    size="lg",
                )

                gr.Markdown("### 💡 Model Answer")
                output_answer = gr.Textbox(
                    label="Prediction", show_label=False, lines=5, interactive=False
                )

        # --- Examples ---
        gr.Markdown("### 🧪 Try with Examples")
        gr.Examples(
            examples=[],  # samples as: ["examples/img.jpg", "question"]
            inputs=[input_img, input_question],
            label="Click an example to load",
        )

        # --- CITATION area ---
        with gr.Accordion("📝 Cite this work", open=False):
            gr.Code(
                value=PROJECT_INFO["citation"],
                language="latex",
                label="BibTeX",
                interactive=False,
            )

        # --- Event ---
        submit_btn.click(
            fn=answer_question,
            inputs=[input_img, input_question],
            outputs=output_answer,
        )
        input_question.submit(
            fn=answer_question,
            inputs=[input_img, input_question],
            outputs=output_answer,
        )

    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    build_app()
