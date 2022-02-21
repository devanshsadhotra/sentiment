
import re
from transformers import TFBertModel, BertConfig, BertTokenizerFast
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
import argparse
import sys
import os

# Name of the BERT model to use
model_name = 'bert-base-uncased'
# Max length of tokens
max_length = 30
sentiment_model = tf.keras.models.load_model('my_model.h5')
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name, config=config)

# def parse_arguments():
#     """
#     It takes a varible to display predicted probabilites of sentinments
#     """
#     parser = argparse.ArgumentParser(description='Sentimental Analyser')
#     parser.add_argument('-p', '--prob', default='False', type=bool, dest='prob' ,help='Diplay probabilities of sentiments')
 
#     if len(sys.argv) == 1:
#         parser.print_help()
#         sys.exit(1)
    
#     return parser.parse_args()

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
      input_text,
        max_length=30, 
        truncation=True, 
        padding='max_length', 
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
    }


def make_prediction(model, processed_data, classes=None):
    # if classes is None:
    #     classes = ['Negative', 'A bit negative', 'Neutral', 'A bit positive', 'Positive']
    probs = model.predict(processed_data)['product'][0]
    return probs
    #return classes[np.argmax(probs)]



app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    val_dict={ 
            'input_text':'',
            'result':'not_predicted_yet',
        }
    if request.method == 'POST':
        form_dict = request.form
        input_text=form_dict['input_text']
        val_dict['input_text']=input_text
        
        if input_text=='' or input_text.isnumeric():
            val_dict['input_text']='No'
            return render_template("home.html",prediction=val_dict)
            
        else:
            processed_data = prepare_data(input_text, tokenizer)
            probs = make_prediction(sentiment_model, processed_data=processed_data)
            classes = ['Negative', 'A bit negative', 'Neutral', 'A bit positive', 'Positive']
            val_dict['result']=classes[np.argmax(probs)]
            return render_template("home.html", prediction=val_dict)

    return render_template("home.html", prediction=val_dict)




if __name__ == "__main__":
    # global args
    # args = parse_arguments()
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
