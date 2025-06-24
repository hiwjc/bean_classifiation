# Gradio UI

import gradio as gr

def classify_image(img):
    classifier = BeanDiseaseClassifier()
    return classifier.predict(img)

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='pil'),
    outputs='text',
    title='Bean Disease Classifier',
    description='Upload an image of a bean leaf to classify the disease.'
)

iface.launch()
