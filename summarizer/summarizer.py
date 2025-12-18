from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load T5-small model for summarization
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize_text(text):
    # Add summarization prefix
    input_text = "summarize: " + text

    # Encode input
    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    # Generate summary
    summary_ids = model.generate(
        inputs,
        max_length=100,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode output
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
