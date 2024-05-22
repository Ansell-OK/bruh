from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

model = tf.saved_model.load("dokinta_model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def classify_text():
    data = request.json 
    text = data['text']
    
    text_tensor = tf.constant([text], shape=(1, 1), dtype=tf.string, name='input_2')
    predictions = model(text_tensor, training=False)
    text_argmax = tf.argmax(predictions, axis =1)
    class_names = ['Common Cold', 'Dengue', 'Malaria', 'Typhoid']
    prediction = class_names[text_argmax[0]]

    print(prediction)
    
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
