import pandas as pd
import numpy as np
import spacy
import os
import gradio as gr
import umap
from sklearn.cluster import OPTICS
from transformers import BertTokenizer, TFBertModel
import plotly.io as pio

# configuration params
pio.templates.default = "plotly_dark"

# setting up the text in the page
TITLE = "<center><h1>BERTopic - For topics detection on text</h1></center>"
DESCRIPTION = r"""<center>Apply BERTopic to a given dataset end extract the most relevant topics.<br>
                 """
EXAMPLES = [
    ["data/ecomm500.csv"],
]
ARTICLE = r"""<center>
              Done by dr. Gabriel Lopez<br>
              This program follows the BERTopic philosophy, but actually has its own implementation.<br>
              For more please visit: <a href='https://sites.google.com/view/dr-gabriel-lopez/home'>My Page</a><br>
              For info about the BERTopic model can be <a href="https://maartengr.github.io/BERTopic/index.html">found here</a><br>
              </center>"""


def load_data(fileobj):
    """Load dataset (keep only 500 rows for efficiency)"""
    data = pd.read_csv(fileobj.name, on_bad_lines='skip', nrows=500)
    assert "text" in data.columns, "The data must have a column named 'text'"
    return data[['text']]


def run_nlp_processing(data):
    """As reference for standard NLP processing"""
    # NLP processing
    docs = []
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    for doc in nlp.pipe(data["text"].values, n_process=os.cpu_count() - 1):
        lemmas = []
        for token in doc:
            if token.is_punct or token.is_stop:
                continue
            lemmas.append(token.lemma_.lower())
        docs.append(" ".join(lemmas))
    # Make new column
    data = data.assign(text=docs)
    return data


def run_bert_tokenization(data):
    """Show the action of the WordPiece alogorithm"""
    # load BERT model (for embeddings)
    checkpoint = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    model = TFBertModel.from_pretrained(checkpoint)
    # Run BERT tokenizing + encoding
    descr_processed_tokenized = tokenizer(
        list(data["text"]),
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128,
    )
    data = data.assign(text_tokenized=descr_processed_tokenized)
    return data


def run_bertopic(data):
    """ " End-to-end BERTopic model"""
    # load BERT model (for embeddings)
    checkpoint = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    model = TFBertModel.from_pretrained(checkpoint)
    # Run BERT tokenizing + encoding
    descr_processed_tokenized = tokenizer(
        list(data["text"]),
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128,
    )
    output_bert = model(descr_processed_tokenized)
    # Get sentence embeddings from BERTs word embeddings
    mean_vect = []
    for vect in output_bert.last_hidden_state:
        mean_vect.append(np.mean(vect, axis=0))
    data = data.assign(descr_vect=mean_vect)
    # Use UMAP to lower the dimensionality of the embedding to 3D - [stack makes array(array()) --> array2d]
    descr_vect_3d = umap.UMAP(n_components=3).fit_transform(
        np.stack(data["descr_vect"].values)
    )
    data["descr_vect_2d"] = list(descr_vect_3d)
    # Use BERT's + UMAP vector embeddings for clustering using OPTICS
    clustering = OPTICS(min_samples=50).fit(np.stack(data["descr_vect_2d"].values))
    data["cluster_label"] = clustering.labels_
    # Plot the 3D embedding
    fig_bertopic = plot_bertopic(descr_vect_3d, data)
    # Extract topic wordclouds
    return fig_bertopic


def plot_bertopic(descr_vect_3d, data):
    """ " Show the topic clusters over an 3d embedding space"""
    import plotly.express as px

    fig = px.scatter_3d(
        x=descr_vect_3d[:, 0],
        y=descr_vect_3d[:, 1],
        z=descr_vect_3d[:, 2],
        color=data["cluster_label"],
    )
    return fig


# gradio interface
blocks = gr.Blocks()
with blocks:
    # physical elements
    session_state = gr.State([])
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "## Load the data (must be a csv file with a column named 'text')"
            )
            in_file = gr.File()
            gr.Markdown("## Inspect the data")
            in_data = gr.Dataframe(max_rows=5)
            submit_button = gr.Button("Run BERTopic!")
            gr.Examples(inputs=in_file, examples=EXAMPLES)
        with gr.Column():
            gr.Markdown("## BERTopic Flow")
            gr.Markdown(
                "Text -> Word-Piece Tokenization -> BERT-embedding -> UMAP -> HDBSCAN -> Topic"
            )
            gr.Markdown("## Processed Text")
            out_dataset = gr.Dataframe(max_rows=5)
            gr.Markdown("## Embedding + Projection + Clustering")
            embedding_plot = gr.Plot(label="BERTopic projections")
            gr.Markdown("## Extracted Topics")
            topics_text = gr.Textbox(label="Topics", lines=50)
    gr.Markdown(ARTICLE)
    # event listeners
    in_file = in_file.upload(inputs=in_file, outputs=in_data, fn=load_data)
    submit_button.click(inputs=in_data, outputs=out_dataset, fn=run_bert_tokenization)
    # out_dataset.change(inputs=out_dataset, outputs=embedding_plot, fn=run_bertopic)

blocks.launch()
