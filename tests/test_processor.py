import os
import random
import torch
import pytest
from transformers import BatchFeature
from PIL import Image
from src.processing_divedoc import get_processor

@pytest.fixture(scope="class")
def processor():
    hf_token = os.getenv("HF_TOKEN")
    return get_processor(hf_token,2048,2048,4096)


class TestProcessor:
    """Test data processing."""
    
    @pytest.mark.parametrize("image_size", [(1000, 1000), (3000, 3000), (150, 500)])
    def test_processor_resizing_image(self,processor,image_size):
        fake_image = Image.new('RGB', image_size, color='black')
        
        inputs = processor(text="Test", images=fake_image, return_tensors="pt",padding=True)
        
        # Inputs check
        assert "input_ids" in inputs
        assert "pixel_values" in inputs
        assert "attention_mask" in inputs

        # Type check
        assert isinstance(inputs,(dict, BatchFeature))
        assert isinstance(inputs["pixel_values"],torch.Tensor)
        assert isinstance(inputs["attention_mask"],torch.Tensor)
        assert isinstance(inputs["input_ids"],torch.Tensor)

        # Dim check 
        assert inputs["attention_mask"].shape[1] == inputs["input_ids"].shape[1] #sequence length 
        assert inputs["attention_mask"].shape[0] == inputs["input_ids"].shape[0] == inputs["pixel_values"].shape[0] == 1 #batch dim 
        assert inputs["pixel_values"].shape[1] == 3 #RGB image tensor
        assert inputs["pixel_values"].shape[2] == processor.image_processor.size['height'] #match target height
        assert inputs["pixel_values"].shape[3] == processor.image_processor.size['width'] #match target width
    

    def test_processor_handles_batches(self,processor):
        small_fake_image = Image.new('RGB', (1000, 1000), color='black')
        image_list = [small_fake_image, small_fake_image] 
        
        inputs = processor(text=["Question 1", "Question two"], images=image_list, return_tensors="pt",padding=True)
        
        # Batch size length check
        assert inputs["pixel_values"].shape[0] == 2
        assert inputs["input_ids"].shape[0] == 2
        assert inputs["attention_mask"].shape[0] == 2

    def test_processor_decode(self,processor):
        original_text = "DIVE-Doc is a 2.5B end-to-end VLM!"
        
        
        encoded = processor.tokenizer(original_text, return_tensors="pt",padding=True)
        input_ids = encoded["input_ids"][0] 
        
        decoded_text = processor.decode(input_ids, skip_special_tokens=True)
        
        # Type check
        assert isinstance(decoded_text, str)

        # Length check
        assert len(decoded_text) > 0
        
        # Content check
        assert original_text == decoded_text