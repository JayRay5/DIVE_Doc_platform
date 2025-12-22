# file and os lib managements
import json
import os
import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Change the path of some HF environment variables to store the download data (model and dataset) from the hub to a choosen location
os.environ["HF_HOME"] = "../.cache"
os.environ["HF_HUB_CACHE"] = "../.cache"
os.environ["HF_DATASETS_CACHE"] = "../.cache"

# ML libraries
from accelerate import infer_auto_device_map, dispatch_model
from datasets import load_dataset
import torch

from modeling_divedoc import get_model
from processing_divedoc import get_processor


def test_results_generation(path):
    # create a new folder to save the results
    results_path = "{}/results".format(path)
    if "results" not in os.listdir(path):
        os.mkdir(results_path)

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("Error: HF_TOKEN not found!")
    """

    STUDENT LOADING

    """

    model = get_model()
    processor = get_processor(hf_token, img_height=2048, img_width=2048, img_lm_input_seq_length=4096)

    """

    Evaluation 


    """

    test_dataset = load_dataset(
        "pixparse/docvqa-single-page-questions", split="test", streaming=True, revision="33136ef456fa5a3fe68568d6e31dda4eeff95b9b"
    )
    batch_dataset = test_dataset.batch(2)

    device_map = infer_auto_device_map(
        model,
        max_memory={d: "5GiB" for d in range(torch.cuda.device_count())},
        no_split_module_classes=["DonutSwinStage", "GemmaDecoderLayer"],
    )
    model = dispatch_model(model, device_map)
    pred_list = []

    print("[INFO] Generate answers on the test set [INFO]")
    with torch.inference_mode():
        for batch in tqdm.tqdm(batch_dataset):
            # preprocessing
            imgs = []
            for i in batch["image"]:
                imgs.append(i.convert("RGB"))
            txt = batch["question"]
            model_inputs = processor(
                text=txt, images=imgs, return_tensors="pt", padding=True
            )
            input_len = model_inputs["input_ids"].shape[-1]

            # processing
            model_inputs = model_inputs.to(model.device)
            generation = model.generate(
                **model_inputs, max_new_tokens=100, do_sample=False
            )

            # postprocessing
            for i in range(len(generation)):
                pred = generation[i][input_len:]
                decoded = processor.tokenizer.decode(pred, skip_special_tokens=True)
                pred_list.append(
                    {"answer": decoded, "questionId": batch["question_id"][i]}
                )
                print(
                    "QuestionId : {}, Question : {}, Reponse : {}".format(
                        batch["question_id"][i], batch["question"][i], decoded
                    )
                )

        results_file = "model_docvqa_test_results.json"
        with open("{}/{}".format(results_path, results_file), "w") as f:
            json.dump(pred_list, f)


if __name__ == "__main__":
    default_path = "./"
    test_results_generation(default_path)
