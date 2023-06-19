from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import privateGPT  # replace with your own module name

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/path/to/upload'  # replace with your upload path
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # TODO: Add logic to process the file and add to your document database
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('upload.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        # TODO: Add logic to query your QA system
        # answers = your_qa_system_module.ask(query)
        # The `answers` object should include the answers, related documents and highlights
        #return render_template('search.html', answers=answers)
    return render_template('search.html', answers=None)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        responses =[]
        try:
            query = request.form['message']
            hide_source = False  # or get this value from somewhere else
            use_openai = request.form.get("use_openai") == "true"  # New line to get the checkbox value

            answer, docs = privateGPT.chatDocument(query, hide_source,use_openai)

            response ={
                'message':query,
                'response':answer,
                'docs':docs
            }
            responses.append(response)
           # responses['message'] =query
           # responses['response'] = answer
           # responses['docs'] = docs

            return render_template('chat.html', responses=responses)
        except Exception as e:
            error_message = "An error occurred: " + str(e)
            return render_template('chat.html', error=error_message)

    return render_template('chat.html')
if __name__ == '__main__':
    app.run(debug=True)
