from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import sys
from googletrans import Translator

# Crear una instancia del traductor
translator = Translator()
argumentos = sys.argv[1:]
model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image = Image.open(argumentos[0])
enc_image = model.encode_image(image)
resultado = model.answer_question(enc_image, "Give a big description of this image in huge detail", tokenizer)
translated = translator.translate(resultado, src='en', dest='es')
print(translated.text)

