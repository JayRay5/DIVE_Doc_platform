import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
#Change the path of some HF environment variables to store the download data (model and dataset) from the hub to a choosen location
os.environ['HF_HOME'] = "../.cache"
os.environ['HF_HUB_CACHE'] = "../.cache"
os.environ['HF_DATASETS_CACHE'] = "../.cache"
import gradio as gr
import torch
from transformers import AutoProcessor, DonutProcessor
from accelerate import infer_auto_device_map, dispatch_model



def app(path):
    model = DIVEdoc.from_pretrained(path)
    with open("./token.json", "r") as f:
            hf_token = json.load(f)["HF_token"]

   
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)
    processor.image_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token).image_processor
    image_input_resolution = model.config.vision_config.encoder_config.image_size
    processor.image_processor.size = {'height': image_input_resolution[0], 'width': image_input_resolution[1]}

    if torch.cuda.is_available():
        device_map = infer_auto_device_map(model,max_memory={0: "6GiB"}, 
                                            no_split_module_classes=["DonutSwinStage","GemmaDecoderLayer"])
            
        model = dispatch_model(model,device_map)
    else: 
         model.cpu()

    def answer_question(image, question):
        # Process the image and question
        model_inputs = processor(text=question, images=image, return_tensors="pt",padding=True)
        model_inputs = model_inputs.to(model.device,dtype=model.dtype)
        input_len = model_inputs["input_ids"].shape[-1]

        # Answer generation
        model.eval()
        with torch.inference_mode():
            pred = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)[0][input_len:]

        answer = processor.decode(pred, skip_special_tokens=True)

        print(f"Question:{question}\nAnswer:{answer}")
        return answer

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
    path = "JayRay5/DIVE-Doc-FRD"
    app(path) 