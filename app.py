import gradio as gr
import torch
from transformers import (
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from PIL import Image # Make sure PIL is imported if not already

# ─── 1. Quantization & Memory Setup ───────────────────────────────────────
print("Setting up quantization configuration...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # bnb_8bit_quant_type="fp4", # 8-bit doesn't use fp4, remove or use load_in_4bit=True if 4-bit is desired
    bnb_8bit_compute_dtype=torch.float16
)
# Example for RTX 3060 with 6 GB VRAM + CPU RAM for offloading
max_mem = {0: "6GB", "cpu": "30GB"}

# ─── 2. Processor, Tokenizer & Model Initialization ───────────────────────
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
TARGET_SIZE = 224  # For consistent image processing
PIXELS = TARGET_SIZE * TARGET_SIZE

print(f"Loading Processor for {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    min_pixels=PIXELS,
    max_pixels=PIXELS
)

print(f"Loading Tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# Add potentially missing special tokens used by templates/generation
# Although apply_chat_template should handle internal tokens, ensuring
# common ones exist can sometimes help compatibility.
# Check tokenizer.added_tokens_decoder to see if they exist already.
print("Checking and adding special tokens...")
special_tokens_to_add = []
tokens_to_check = ["<|begin_of_sentence|>", "<img>", "</img>", "<tool_response>", "USER:", "ASSISTANT:"]
for token in tokens_to_check:
    if token not in tokenizer.get_vocab():
         special_tokens_to_add.append(token)

if special_tokens_to_add:
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    print(f"Added special tokens: {special_tokens_to_add}")
else:
    print("Required special tokens seem to exist.")

# Ensure pad token is set if it's None
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")


print(f"Loading Model {MODEL_NAME} with quantization...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto", # Automatically distribute layers based on max_memory
    max_memory=max_mem,
    offload_folder="offload", # Specify folder for offloaded layers
    use_safetensors=True,
    trust_remote_code=True
)

# Resize token embeddings if new tokens were added
print("Resizing token embeddings if necessary...")
model.resize_token_embeddings(len(tokenizer))

print("Setting model to evaluation mode...")
model.eval() # Set model to evaluation mode

# ─── 3. Chat History Storage ──────────────────────────────────────────────
chat_history = []

# ─── 4. Inference Function ────────────────────────────────────────────────
def answer_question(image, question):
    global chat_history # Access the global chat history

    if image is None:
        # Return current history and an error message
        display = [(f"User: {q}", f"Assistant: {a}") for q, a in chat_history]
        return display, "Please upload an image."
    if not question or question.strip() == "":
        # Return current history and an error message
        display = [(f"User: {q}", f"Assistant: {a}") for q, a in chat_history]
        return display, "Please ask a question."

    # Ensure image is PIL Image and in RGB format, then resize
    if not isinstance(image, Image.Image):
         # If input is numpy array or other format, convert to PIL
         try:
             image = Image.fromarray(image)
         except Exception as e:
             print(f"Error converting image input to PIL: {e}")
             display = [(f"User: {q}", f"Assistant: {a}") for q, a in chat_history]
             return display, "Invalid image format."

    try:
        image = image.convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
    except Exception as e:
        print(f"Error processing image (convert/resize): {e}")
        display = [(f"User: {q}", f"Assistant: {a}") for q, a in chat_history]
        return display, "Error processing image."


    # --- Use apply_chat_template for Robust Input Formatting ---
    # Create the message structure expected by the template
    messages = [
        {"role": "user", "content": [
            {"type": "image"}, # Placeholder for the image
            {"type": "text", "text": question.strip()} # Use stripped question
        ]}
    ]

    try:
        # Apply the template to get the formatted text string with image placeholders
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # Instructs model to generate a response
        )
    except Exception as e:
        print(f"Error applying chat template: {e}")
        # Fallback or error message
        display = [(f"User: {q}", f"Assistant: {a}") for q, a in chat_history]
        return display, "Error formatting prompt."


    # Preprocess vision + text using the template-formatted text
    try:
        inputs = processor(
            text=text,         # Use the formatted text from the template
            images=[image],    # Provide the actual image
            return_tensors="pt"
        ).to(model.device)  # Move tensors to the correct device
    except Exception as e:
         print(f"Error during processor call: {e}")
         display = [(f"User: {q}", f"Assistant: {a}") for q, a in chat_history]
         return display, "Error processing inputs for model."


    # Generate response using greedy search (do_sample=False)
    try:
        print("Generating response...")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,  # Max length of the generated response
            do_sample=False,     # Use greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id # Use model's configured EOS or specific list
            # You might need a list for eos_token_id if multiple tokens signify end, e.g.:
            # eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        )
        print("Generation complete.")
    except Exception as e:
        print(f"Error during model generation: {e}")
        display = [(f"User: {q}", f"Assistant: {a}") for q, a in chat_history]
        return display, "Error generating response from model."


    # Decode only the newly generated tokens
    try:
        input_token_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_token_len:]
        # Use processor.decode for potentially better handling of special tokens
        answer = processor.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"Decoded Answer: {answer}")
    except Exception as e:
        print(f"Error during decoding: {e}")
        answer = "[Decoding Error]"


    # Update chat history
    chat_history.append((question, answer))
    display = [(f"User: {q}", f"Assistant: {a}") for q, a in chat_history]

    # Return the updated chat display and clear the temporary error message box
    return display, ""

# ─── 5. Gradio Interface ──────────────────────────────────────────────────
print("Setting up Gradio interface...")
with gr.Blocks() as demo:
    gr.Markdown(f"## visionary Q&A ")
    gr.Markdown("Upload an image and ask a question about it.")
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Image") # Use PIL format
            txt_input = gr.Textbox(label="Your question")
            ask_button = gr.Button("Ask")
        with gr.Column(scale=2):
            chat_output = gr.Chatbot([], label="Chat History", elem_id="chatbot") # Use Chatbot component
            error_box = gr.Textbox(label="Status/Error", visible=True, interactive=False) # To show status/errors

    # Link button click to function
    ask_button.click(
        fn=answer_question,
        inputs=[img_input, txt_input],
        outputs=[chat_output, error_box] # Output to chatbot and error box
    )
    # Clear inputs on submit
    ask_button.click(lambda: (None, ""), outputs=[img_input, txt_input])


if __name__ == "__main__":
    print("Launching Gradio demo...")
    demo.queue() # Enable queue for handling multiple users/requests
    demo.launch(share=False) # Set share=True to get a public link (use with caution)
    print("Gradio demo launched.")