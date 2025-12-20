import os
import pytest
from src.modeling_divedoc import get_model
from src.processing_divedoc import get_processor

@pytest.fixture(scope="session")
def hf_token():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        pytest.fail("Error, the huggingface token has not been find in the HF_TOKEN env variable!")
    return hf_token

@pytest.fixture(scope="session")
def processor(hf_token):
    return get_processor(hf_token,2048,2048,4096)

@pytest.fixture(scope="session")
def model():
    model = get_model()
    assert model.training == False
    assert all(not p.requires_grad for p in model.parameters())
    return model