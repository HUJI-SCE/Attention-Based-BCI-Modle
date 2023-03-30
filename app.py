from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    
    # check if getting a name from the query string "e.g. ?name=your_name"
    if request.args.get('name'):
        user_name = request.args.get('name')
    else:
        user_name = "Student Name"

    # render the template with the user name
    return render_template("index.html", title="Hello", username=user_name)
 
