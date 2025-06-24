# 실전 적용 (추론)

import torch
from PIL import Image
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)
from bean_model import model  # 사용자 정의한 bean_model 임포트

class BeanDiseaseClassifier:
    def __init__(self):
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = model.load_model()  # bean_model 내부의 load_model 메소드를 호출

    def predict(self, image_path):
        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        predicted_class = probs.argmax().item()
        label_name = self.model.config.id2label[str(predicted_class)]
        return label_name

### 
    class BeanDiseaseClassifier:
    def __init__(self):
        self.image_processor = image_processor
        self.model = model

    def predict(self, image):
        inputs = self.image_processor(images=image, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        label_id = probs.argmax().item()
        return self.model.config.id2label[str(label_id)]
