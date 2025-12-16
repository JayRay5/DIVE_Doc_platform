import sys
from pathlib import Path
parent_root = Path().resolve().parent.parent 
sys.path.append(str(parent_root))




from transformers import PretrainedConfig, DonutSwinConfig, GemmaConfig, CONFIG_MAPPING, SiglipVisionConfig
from typing import Tuple, Literal



class PamConfig(PretrainedConfig): 
     model_type = "pam"
     def __init__(
        self,
        sequence_mapping_layer_type: Literal["linear_projection","bilinear_interpolation"] = "bilinear_interpolation",
        student_fmap_dim: Tuple[int,int]=(80,60),
        student_embedding_dim: int = 1024,
        teacher_fmap_dim: Tuple[int,int] = (64,64),
        teacher_embedding_dim: int = 1152,
        **kwargs,
    ):
        self.sequence_mapping_layer_type = sequence_mapping_layer_type
        self.student_fmap_dim = student_fmap_dim
        self.student_embedding_dim = student_embedding_dim
        self.teacher_fmap_dim = teacher_fmap_dim
        self.teacher_embedding_dim = teacher_embedding_dim
        super().__init__(**kwargs)


class SwinPamVisionEncoderConfig(PretrainedConfig): 
    model_type = "swinpam"
    sub_configs = {"encoder_config": DonutSwinConfig, "pam_config": PamConfig}
    def __init__(
        self,
        encoder_config: DonutSwinConfig = None,
        pam_config: PamConfig = None,
        **kwargs
    ):
        self.encoder_config = encoder_config
        self.pam_config = pam_config

        if isinstance(self.encoder_config, dict):
            encoder_config["model_type"] = (
                encoder_config["model_type"] if "model_type" in encoder_config else "donut-swin"
            )
            if encoder_config["model_type"] == "donut-swin":
                self.encoder_config = DonutSwinConfig(**encoder_config)
            else:
                print(f"Encoder type: {encoder_config['model_type']}")
                self.encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        
        '''
        elif encoder_config is None:
            print("coucou2")
            self.encoder_config = DonutSwinConfig()
        '''

        if isinstance(self.pam_config, dict):
            '''
            pam_config["model_type"] = (
                pam_config["model_type"] if "model_type" in pam_config else "pam"
            )
            '''
            if pam_config["model_type"] == "pam":
                self.pam_config = PamConfig(**pam_config)
            else:
                raise ValueError(f"pam_config['model_type'] should be 'pam', got {pam_config['model_type']}")
        '''
        elif pam_config is None:
            self.pam_config = PamConfig()
        '''
        super().__init__(**kwargs)


class DIVEdocConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"vision_config": SwinPamVisionEncoderConfig, "text_config": GemmaConfig}
    model_type = "DIVEdoc"
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        #_attn_implementation_autoset = True,
        **kwargs,
    ):
        self._ignore_index = ignore_index
        self.image_token_index = image_token_index
        self._vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        #self._attn_implementation_autoset = _attn_implementation_autoset
    

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "swinpam"
            )
            if vision_config["model_type"] == "swinpam":
                self.vision_config = SwinPamVisionEncoderConfig(encoder_config=vision_config["encoder_config"],pam_config=vision_config["pam_config"])
            elif vision_config["model_type"] == "siglippam":
                self.vision_config = SiglipPAMVisionEncoderConfig(encoder_config=vision_config["encoder_config"],pam_config=vision_config["pam_config"])
            else:
                self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = get_vision_config("swinpam")

        self.text_config = text_config
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "gemma"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["gemma"](
                hidden_size=2048,
                num_hidden_layers=18,
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
                is_encoder_decoder=False,
                vocab_size=vocab_size,
            )
    
        self.text_config.num_image_tokens = self.vision_config.pam_config.teacher_fmap_dim[0] *\
                                            self.vision_config.pam_config.teacher_fmap_dim[1]
        self.vision_config.projection_dim = projection_dim
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output.pop("_ignore_index", None)
        return output

def get_siglip_vision_config(image_size=[896,896],num_image_token = 4096,hidden_size = 768):
    encoder_config = SiglipVisionConfig(
                                        hidden_size = hidden_size,
                                        image_size = image_size,
                                        intermediate_size = 2860,
                                        model_type = "siglip_vision_model",
                                        num_attention_heads = 8,
                                        num_hidden_layers = 12,
                                        num_image_tokens = num_image_token,
                                        patch_size = 14,
                                        projection_dim = 2048,
                                        projector_hidden_act = "gelu_fast",
                                        torch_dtype = "float32",
                                        vision_use_head = False
                                    )
    return encoder_config

def get_swin_vision_config(image_size=[2560,1920],hidden_size = 1024):
    encoder_config = DonutSwinConfig(
        attention_probs_dropout_prob= 0.0,
        depths =[
            2,
            2,
            14,
            2
        ],
        drop_path_rate= 0.1,
        embed_dim =128,
        hidden_act ="gelu",
        hidden_dropout_prob = 0.0,
        hidden_size = hidden_size,
        image_size = image_size,
        initializer_range = 0.02,
        layer_norm_eps = 1e-05,
        mlp_ratio = 4.0,
        model_type = "donut-swin",
        num_channels = 3,
        num_heads =[
            4,
            8,
            16,
            32
        ],
        num_layers =4,
        patch_size = 4,
        path_norm = True,
        qkv_bias = True,
        use_absolute_embeddings = False,
        window_size = 10
        )
    return encoder_config

def get_vision_config(  visual_encoder_type:Literal["swinpam","siglip80m"],
                        image_size=[2560,1920],
                        sequence_mapping_layer_type= "bilinear",
                        student_fmap_dim=(80,60),
                        student_embedding_dim= 1024,
                        teacher_fmap_dim= (64,64),
                        teacher_embedding_dim= 1152):
    pam_config = PamConfig(
                    sequence_mapping_layer_type = sequence_mapping_layer_type,
                    student_fmap_dim = student_fmap_dim,
                    student_embedding_dim = student_embedding_dim,
                    teacher_fmap_dim = teacher_fmap_dim,
                    teacher_embedding_dim = teacher_embedding_dim)
    
    if visual_encoder_type == "swinpam":
        encoder_config = get_swin_vision_config(image_size=image_size,hidden_size = student_embedding_dim)
        ve_config = SwinPamVisionEncoderConfig(encoder_config=encoder_config,pam_config=pam_config)
        return ve_config

    else:
        raise ValueError(f"Unknown visual encoder type, need 'swinpam', got {visual_encoder_type}.")
