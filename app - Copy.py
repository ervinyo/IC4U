from flask import Flask, jsonify, render_template, request
import traceback
from flask_restful import Resource, Api
import webbrowser as wb
app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        #return {'hello': 'world'}
        #wb.open('http://127.0.0.1:5000/')
        return render_template('home.html')
        #return "http://127.0.0.1:5000/"

#api.add_resource(HelloWorld, '/')
@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/Youtube/')
def youtube():
    return render_template('youtube.html')	
def index2():
    return "hello world!"
@app.route('/test', methods=['POST'])
def test():
    clicked=None
    if request.method == "POST":
        clicked=request.json['data']
    return "12"
@app.route('/ajax1', methods=['GET', 'POST'])
def ajax1():
    try:
        user =  request.form.get('username')
        return "1"
    except Exception:
        return "error"
		
@app.route('/openapp', methods=['GET', 'POST'])	
def openapp():
    try:
        f = open ('result.txt','r')
        #print (f.read())
        return f.read()
    except Exception:
        return "error"
		
if __name__ == '__main__':
   app.run(debug = True)
