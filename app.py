from flask import Flask, request, jsonify, render_template
from langchain_ollama import ChatOllama
import markdown

app = Flask(__name__)

# Endpoint để trả về trang HTML
@app.route('/')
def index():
    return render_template('index.html')  # Đảm bảo file index.html có trong thư mục "templates"

# Endpoint để gọi mô hình
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json.get('input')

    # Kết nối mô hình
    model = ChatOllama(model="mistral:7b", base_url="http://localhost:11434")

    # Định dạng đúng của messages
    messages = [
        {"role": "system", "content": "You are my Italian teacher."},
        {"role": "user", "content": input_data}
    ]

    # Gọi mô hình
    result = model.invoke(messages)
    html_content = markdown.markdown(result.content)

    # Trả về kết quả JSON
    return jsonify({"result": html_content})  # Lấy nội dung từ .content

if __name__ == '__main__':
    app.run(debug=True)
