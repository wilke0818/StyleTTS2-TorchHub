import os
from typing import Literal
import torch
import yaml

from inference import StyleTTS2


dependencies = ['librosa', 
                'numpy', 
                'phonemizer', 
                'torch', 
                'torchaudio', 
                'yaml', 
                'nltk',
                'monotonic_align'
                ]


libri_tts_url = 'https://github.com/wilke0818/StyleTTS2-TorchHub/releases/download/v1.0.0-alpha.1/epochs_2nd_00020.pth'
ljspeech_url = 'https://github.com/wilke0818/StyleTTS2-TorchHub/releases/download/v1.0.3/LJSpeech.pth'

def styletts2(progress=True, device=None, 
              pretrain_data: Literal['LibriTTS', 'LJSpeech'] ='LibriTTS') -> StyleTTS2:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = yaml.safe_load(open(f"{dir_path}/Models/{pretrain_data}/config.yml"))
    ckpt_path = libri_tts_url if pretrain_data == 'LibriTTS' else ljspeech_url
    params_whole = torch.hub.load_state_dict_from_url(ckpt_path, progress=progress, check_hash=False, map_location="cpu")
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return StyleTTS2(config=config, params_whole=params_whole, dir_path=dir_path, device=device)

