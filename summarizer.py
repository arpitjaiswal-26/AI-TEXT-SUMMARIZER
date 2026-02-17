import os, warnings
import PyPDF2

import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

warnings.filterwarnings("ignore")

# ------------------------- DEVICE -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 150

# ------------------------- LOAD MODEL -------------------------
print("\nLoading FLAN-T5 model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

#                SUMMARY MODES / PROMPTS
 
SUMMARY_MODES = {
    "short":   "give a short concise summary: ",
    "long":    "give a detailed multi-paragraph summary: ",
    "bullets": "summarize as clear bullet points: ",
    "normal":  "summarize this clearly: "
}

CURRENT_MODE = "normal"

def read_multiline_input():
    print("\nPaste your text below.")
    print("When finished, press ENTER again on an empty line:\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    return
def summarize_text_block(text: str) -> str:
    """Summarize a text block using the selected mode."""
    prompt = SUMMARY_MODES[CURRENT_MODE] + text

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        padding="max_length"
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=MAX_OUTPUT_LEN,
            num_beams=4,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def summarize(text: str) -> str:
    """Handles short + long text automatically using chunking."""
    text = text.strip()

    if len(text) == 0:
        return "❌ No content found."

    if len(text) < 500:
        return summarize_text_block(text)

    chunks = []
    step = 1500

    for i in range(0, len(text), step):
        chunks.append(text[i:i+step])

    summaries = [summarize_text_block(ch) for ch in chunks]

    final_text = " ".join(summaries)
    final_summary = summarize_text_block(final_text)

    return final_summary


#                        PDF READER

def read_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                content = page.extract_text()
                if content:
                    text += f"\n\n--- PAGE {i+1} ---\n\n"
                    text += content
            except:
                pass
    return text


#                Summarize ONE BBC Article ONLY

def summarize_single_bbc_article():
    print("\nAvailable categories: business, tech, sport, politics, entertainment\n")

    cat = input("Enter category: ").strip().lower()
    file = input("Enter article filename (example: 001.txt): ").strip()

    base_path = r"C:\Users\arpit\OneDrive\Desktop\AIML PROJECT\BBC News Summary\BBC News Summary\News Articles"
    path = os.path.join(base_path, cat, file)

    if not os.path.exists(path):
        print("\n❌ Article not found. Check filename/category.\n")
        return

    print("\nReading article...")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print("\nSUMMARY:\n")
    print(summarize(text))


#                          MENU

def menu():
    global CURRENT_MODE

    while True:
        print("\n=====================================")
        print("          UNIVERSAL SUMMARIZER")
        print("=====================================")
        print(f"Current Summary Mode: {CURRENT_MODE.upper()}")
        print("-------------------------------------")
        print("1) Summarize Text")
        print("2) Summarize PDF")
        print("3) Summarize ONE BBC Article")
        print("4) Change Summary Mode")
        print("5) Exit")
        
        choice = input("\nEnter choice: ")

        # ------------ TEXT ------------
        if choice == "1":
            text = read_multiline_input()
            print("\nSUMMARY:\n", summarize(text))

        # ------------ PDF ------------
        elif choice == "2":
            pdf_name = input("\nEnter PDF filename (in same folder): ")
            pdf_path = os.path.join(os.getcwd(), pdf_name)

            if not os.path.exists(pdf_path):
                print("❌ File not found.")
                continue

            print("Reading PDF...")
            text = read_pdf(pdf_path)
            print("\nSUMMARY:\n", summarize(text))

        # ------------ ONE BBC ARTICLE ------------
        elif choice == "3":
            summarize_single_bbc_article()

        # ------------ CHANGE MODE ------------
        elif choice == "4":
            print("\nChoose Summary Mode:")
            print("1) Short")
            print("2) Detailed")
            print("3) Bullet Points")
            print("4) Normal")

            m = input("Enter choice: ")

            if m == "1": CURRENT_MODE = "short"
            elif m == "2": CURRENT_MODE = "long"
            elif m == "3": CURRENT_MODE = "bullets"
            elif m == "4": CURRENT_MODE = "normal"
            else:
                print("❌ Invalid mode.")
                continue

            print("✅ Mode changed to:", CURRENT_MODE.upper())

        # ------------ EXIT ------------
        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("❌ Invalid choice.")


#                        MAIN

if __name__ == "__main__":

    menu()

