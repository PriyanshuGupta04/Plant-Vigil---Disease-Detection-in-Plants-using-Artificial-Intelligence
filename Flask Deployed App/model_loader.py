# import os
# import torch
# import gdown
# from CNN import CNN

# MODEL_PATH = "models/plant_disease_model_1_latest.pt"
# MODEL_URL = "https://drive.google.com/uc?id=18uA9ADFZT-1d1uc7Y9LP68u-0YQKJ2WT"

# def load_model():

#     if not os.path.exists(MODEL_PATH):

#         print("Downloading model...")
#         os.makedirs("models", exist_ok=True)

#         gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

#         print("Download complete.")

#     model = CNN(39)   # IMPORTANT

#     state_dict = torch.load(MODEL_PATH, map_location="cpu")

#     model.load_state_dict(state_dict)

#     model.eval()

#     return model


import os
import torch
import gdown
from CNN import CNN

MODEL_PATH = "models/plant_disease_model_1_latest.pt"
MODEL_URL = "https://drive.google.com/uc?id=18uA9ADFZT-1d1uc7Y9LP68u-0YQKJ2WT"

def load_model():

    if not os.path.exists(MODEL_PATH):

        print("Downloading model...")

        os.makedirs("models", exist_ok=True)

        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

        print("Download complete.")

    # Create architecture
    model = CNN(39)

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()

    return model
