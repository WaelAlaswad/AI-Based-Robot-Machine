from flask import Flask, render_template, url_for
import os



app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/")
def style():
    return render_template("style.css")

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)
def dated_url_for(endpoint, **values):
    if endpoint =='static':
        filename = values.get('filname', None)
        if filename:
            file_path = os.path.join(app.root_path, endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
        return url_for(endpoint, **values)

if __name__ == "__main__":
    app.run(debug=True)



   

