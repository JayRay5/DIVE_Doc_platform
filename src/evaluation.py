#file and os lib managements
import json
import os
import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
#Change the path of some HF environment variables to store the download data (model and dataset) from the hub to a choosen location
os.environ['HF_HOME'] = "../.cache"
os.environ['HF_HUB_CACHE'] = "../.cache"
os.environ['HF_DATASETS_CACHE'] = "../.cache"

#ML libraries
from accelerate import infer_auto_device_map, dispatch_model
from transformers import PaliGemmaProcessor, AutoTokenizer, DonutImageProcessor
from processing_divedoc import PaliGemmaProcessor
from datasets import load_dataset
import torch

from modeling_divedoc import DIVEdoc



def test(path):
    #create a new folder to save the results
    results_path = "{}/results".format(path)
    if "results" not in os.listdir(path):
        os.mkdir(results_path)


    '''

    STUDENT LOADING

    '''

    model = DIVEdoc.from_pretrained( "JayRay5/DIVE-Doc-FRD", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-ft-docvqa-896")
    image_processor = DonutImageProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    image_processor.image_seq_length = 4096
    image_processor.size["height"], image_processor.size["width"] = model.config.vision_config.encoder_config.image_size[0], model.config.vision_config.encoder_config.image_size[1]
    processor = PaliGemmaProcessor(tokenizer=tokenizer, image_processor=image_processor)
    
    """

    Evaluation 


    """

    test_dataset = load_dataset("pixparse/docvqa-single-page-questions", split="test",streaming=True)
    batch_dataset = test_dataset.batch(2) 

    device_map = infer_auto_device_map(model,max_memory={d:"5GiB" for d in range(torch.cuda.device_count())}, 
                                       no_split_module_classes=["DonutSwinStage","GemmaDecoderLayer"])
    model = dispatch_model(model,device_map)
    pred_list = []




    print("[INFO] Generate answers on the test set [INFO]")
    with torch.inference_mode():
        for batch in tqdm.tqdm(batch_dataset):
            #preprocessing 
            imgs = []
            for i in batch["image"]:
                imgs.append(i.convert('RGB'))
            txt = batch["question"]
            model_inputs = processor(text=txt, images=imgs, return_tensors="pt",padding=True)
            input_len = model_inputs["input_ids"].shape[-1]

            #processing
            model_inputs = model_inputs.to(model.device)
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)

            #postprocessing 
            for i in range(len(generation)):
                pred = generation[i][input_len:]
                decoded = processor.tokenizer.decode(pred, skip_special_tokens=True)
                pred_list.append({'answer': decoded, 'questionId': batch['question_id'][i]})
                print("QuestionId : {}, Question : {}, Reponse : {}".format(batch["question_id"][i],batch["question"][i],decoded))


        
        results_file = "model_docvqa_test_results.json"
        with open('{}/{}'.format(results_path,results_file), 'w') as f:
            json.dump(pred_list, f)



if __name__== "__main__":
    default_path = "./"
    test(default_path)
