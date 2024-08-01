
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    print("Hello World")
    return render_template('index.html')

def hello():
    print("HELLO: ", a)

if __name__ == '__main__':
    a = 3
    hello()
    # app.run(host='0.0.0.0', port=5100)
    
