import gradio as gr
import requests
import io

API_URL = "http://127.0.0.1:8000/ask"
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

    interface = gr.Interface(
        fn=answer_question,
        inputs=[gr.Image(type="pil"), gr.Textbox(label="Question")],
        outputs=gr.Textbox(label="Answer"),
        title="Visual Question Answering",
        description="Upload an image of document and ask a question related to the image. The model will try to answer it. \nNote: Processing time depends on whether you’re running the model on a CPU or a GPU."
    ).queue()

    interface.launch( server_name="0.0.0.0",
            server_port=7860)
    
if __name__ == "__main__":
    app() 