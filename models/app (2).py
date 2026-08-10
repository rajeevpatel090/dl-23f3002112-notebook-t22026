import gradio as gr
import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from modeling_cnn import TextCNN
import spaces

# ================================
# LOAD CNN MODEL
# ================================
cnn_model_id = "rajeev5944/cnn-mcq-model"
cnn_model = TextCNN.from_pretrained(cnn_model_id)
cnn_model.eval()
cnn_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

@spaces.GPU
def solve_mcq_cnn(question, option_a, option_b, option_c, option_d, option_e):
    options = [option_a, option_b, option_c, option_d, option_e]
    scores = []
    with torch.no_grad():
        for opt in options:
            if not opt.strip():
                scores.append(float('-inf'))
                continue
            text = f"{question} [SEP] {opt}"
            inputs = cnn_tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
            score = cnn_model(inputs["input_ids"])
            scores.append(score.item())
            
    best_idx = scores.index(max(scores))
    labels = ["A", "B", "C", "D", "E"]
    result = f"Best Answer: Option {labels[best_idx]}\n\nScores:\n"
    for label, opt, score in zip(labels, options, scores):
        result += f"Option {label} ({opt}): {score:.4f}\n"
    return result

# ================================
# LOAD DISTILBERT MODEL
# ================================
distilbert_model_id = "rajeev5944/distilbert-mcq-model"
distilbert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_model_id)
distilbert_model.eval()
distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_id)

@spaces.GPU
def solve_mcq_distilbert(question, option_a, option_b, option_c, option_d, option_e):
    input_text = f"{question} [SEP] {option_a} [SEP] {option_b} [SEP] {option_c} [SEP] {option_d} [SEP] {option_e}"
    with torch.no_grad():
        inputs = distilbert_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        outputs = distilbert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits[0].tolist() 
        
    labels = ["A", "B", "C", "D", "E"]
    options = [option_a, option_b, option_c, option_d, option_e]
    best_idx = logits.index(max(logits))
    result = f"Best Answer: Option {labels[best_idx]}\n\nScores (Logits):\n"
    for label, opt, score in zip(labels, options, logits):
        result += f"Option {label} ({opt}): {score:.4f}\n"
    return result


# ================================
# GRADIO INTERFACE
# ================================
with gr.Blocks() as iface:
    gr.Markdown("# MCQ Solver 😹")
    gr.Markdown("Choose between the CNN and DistilBERT models to solve multiple-choice questions.")
    
    with gr.Tab("CNN Model"):
        with gr.Row():
            with gr.Column():
                q_cnn = gr.Textbox(lines=3, label="Question")
                a_cnn = gr.Textbox(lines=1, label="Option A")
                b_cnn = gr.Textbox(lines=1, label="Option B")
                c_cnn = gr.Textbox(lines=1, label="Option C")
                d_cnn = gr.Textbox(lines=1, label="Option D")
                e_cnn = gr.Textbox(lines=1, label="Option E (Optional)")
                submit_cnn = gr.Button("Solve with CNN")
            with gr.Column():
                output_cnn = gr.Textbox(label="Result")
        submit_cnn.click(solve_mcq_cnn, inputs=[q_cnn, a_cnn, b_cnn, c_cnn, d_cnn, e_cnn], outputs=output_cnn)

    with gr.Tab("DistilBERT Model"):
        with gr.Row():
            with gr.Column():
                q_db = gr.Textbox(lines=3, label="Question")
                a_db = gr.Textbox(lines=1, label="Option A")
                b_db = gr.Textbox(lines=1, label="Option B")
                c_db = gr.Textbox(lines=1, label="Option C")
                d_db = gr.Textbox(lines=1, label="Option D")
                e_db = gr.Textbox(lines=1, label="Option E (Optional)")
                submit_db = gr.Button("Solve with DistilBERT")
            with gr.Column():
                output_db = gr.Textbox(label="Result")
        submit_db.click(solve_mcq_distilbert, inputs=[q_db, a_db, b_db, c_db, d_db, e_db], outputs=output_db)

if __name__ == "__main__":
    iface.launch()
