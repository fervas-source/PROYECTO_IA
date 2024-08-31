import os
import pytesseract
from PIL import Image
import openai

class MedicalImageProcessor:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key

    def extract_text_from_image(self, image_path):
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text

    def analyze_text_with_llm(self, text):
        response = openai.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from medical prescriptions."},
                {"role": "user", "content": f"Extract the medicine name, dosage, and duration from the following text:\n\n{text}\n\nReturn the information in this format:\nMedicine: [medicine name]\nDosage: [dosage]\nDuration: [duration]"}
            ],
            max_tokens=100,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def process_medical_image(self, image_path):
        text = self.extract_text_from_image(image_path)
        extracted_info = self.analyze_text_with_llm(text)
        return extracted_info

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    processor = MedicalImageProcessor(openai_api_key)

    # Ask the user for the image path
    image_path = input("Enter the path to the medical image: ")

    extracted_info = processor.process_medical_image(image_path)
    print(extracted_info)