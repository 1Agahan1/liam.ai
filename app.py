from flask import Flask, request, jsonify, render_template_string
# Import the AI bot class
from import_re import EnhancedLearningQABot

app = Flask(__name__)

# Instantiate the bot once
bot = EnhancedLearningQABot()

@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Liam AI Web UI</title>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%);
                margin: 0;
                padding: 0;
                min-height: 100vh;
            }
            h1 {
                text-align: center;
                color: #2d3a4b;
                margin-top: 40px;
                margin-bottom: 30px;
                letter-spacing: 1px;
            }
            #ai-form {
                background: #fff;
                max-width: 420px;
                margin: 0 auto;
                padding: 32px 28px 24px 28px;
                border-radius: 18px;
                box-shadow: 0 4px 24px rgba(44, 62, 80, 0.08);
                display: flex;
                flex-direction: column;
                gap: 18px;
            }
            #user-input {
                padding: 12px 14px;
                border: 1px solid #cfd8dc;
                border-radius: 8px;
                font-size: 1.1em;
                outline: none;
                transition: border 0.2s;
            }
            #user-input:focus {
                border: 1.5px solid #5b9df9;
            }
            label {
                display: flex;
                align-items: center;
                font-size: 1em;
                color: #3a4a5d;
                gap: 8px;
            }
            button {
                background: linear-gradient(90deg, #5b9df9 0%, #3a8dde 100%);
                color: #fff;
                border: none;
                border-radius: 8px;
                padding: 12px 0;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(44, 62, 80, 0.07);
                transition: background 0.2s, transform 0.1s;
            }
            button:hover {
                background: linear-gradient(90deg, #3a8dde 0%, #5b9df9 100%);
                transform: translateY(-2px) scale(1.03);
            }
            #result {
                max-width: 420px;
                margin: 32px auto 0 auto;
                background: #fff;
                border-radius: 14px;
                box-shadow: 0 2px 12px rgba(44, 62, 80, 0.07);
                padding: 22px 20px;
                font-size: 1.08em;
                color: #2d3a4b;
                min-height: 40px;
                word-break: break-word;
                white-space: pre-line;
            }
            @media (max-width: 600px) {
                #ai-form, #result {
                    max-width: 98vw;
                    padding: 16px 6vw;
                }
                h1 {
                    font-size: 1.3em;
                }
            }
        </style>
    </head>
    <body>
        <h1>Liam AI Web Interface</h1>
        <form id="ai-form">
            <input type="text" id="user-input" placeholder="Enter your input" required>
            <button type="submit">Submit</button>
        </form>
        <div id="result"></div>
        <script>
        document.getElementById('ai-form').onsubmit = async function(e) {
            e.preventDefault();
            const userInput = document.getElementById('user-input').value;
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({input: userInput, use_gemini: true})
            });
            const data = await response.json();
            document.getElementById('result').innerText = data.message || JSON.stringify(data);
        }
        </script>
    </body>
    </html>
    """)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('input') if data else None
    # Always use Gemini, ignore use_gemini from request
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    response = bot.ask_gemini(user_input)
    return jsonify({"message": response})

if __name__ == '__main__':
    app.run(debug=True)