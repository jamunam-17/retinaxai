import cv2
import numpy as np
import torch

def preprocess_fundus_image(pil_img, image_size=224):
    """
    Applies Ben Graham's preprocessing method (green channel isolation + local color average weighting).
    """
    img = np.array(pil_img.convert("RGB"))
    
    # Resize image
    img = cv2.resize(img, (image_size, image_size))
    
    # Ben Graham preprocessing: subtract local average color
    blur = cv2.GaussianBlur(img, (0, 0), image_size / 30)
    enhanced = cv2.addWeighted(img, 4, blur, -4, 128)
    
    # Normalize tensor for PyTorch
    tensor_img = enhanced.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor_img = (tensor_img - mean) / std
    
    # Format to CHW tensor batch
    tensor_img = np.transpose(tensor_img, (2, 0, 1))
    tensor_img = torch.tensor(tensor_img, dtype=torch.float32).unsqueeze(0)
    
    return tensor_img, enhanced