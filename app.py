import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPTNeoForCausalLM
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import openai
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import uuid
import json
from flask import send_file, after_this_request
from datetime import datetime

matplotlib.use('Agg')  # Use non-GUI Agg backend for matplotlib

app = Flask(__name__)
openai.api_key = 'your key'

USER_DATA_DIR = 'user_data'
# Ensure the directory exists
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

# Get the current directory to handle images
current_dir = os.path.dirname(os.path.abspath(__file__))


# Initialize conversation history
conversation_history = []

# Initialize tokenizers and models for each model type
models = {
    'gpt2': {
        'tokenizer': AutoTokenizer.from_pretrained('gpt2'),
        'model': AutoModelForCausalLM.from_pretrained('gpt2')
    },
    'gpt2-medium': {
        'tokenizer': AutoTokenizer.from_pretrained('gpt2-medium'),
        'model': AutoModelForCausalLM.from_pretrained('gpt2-medium')},
    # }
    'gpt-neo': {
        'tokenizer': AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B'),
        'model': GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    }


}

def save_user_interaction(user_id, interaction_type, input_text, model, temperature, temp_slider, p_value, seed,
                          predicted_tokens=None, predicted_probs=None, selected_word=None, selection_method="typed"):
    """Save user interactions with selection method (typed, clicked, auto-selected)."""

    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")

    # Load existing data
    if os.path.exists(user_file):
        with open(user_file, "r") as f:
            user_data = json.load(f)
    else:
        user_data = {"user_id": user_id, "interactions": []}

    # Store the interaction
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "interaction_type": interaction_type,  # 'typed', 'clicked', 'auto-selected'
        "selection_method": selection_method,  # 'typed', 'clicked', 'auto-selected'
        "input_text": input_text,
        "model": model,
        "temperature": temperature,
        "chart_temperature": temp_slider,
        "p_value": p_value,
        "seed": seed,
        "selected_word": selected_word,  # If a word was selected
        "suggested_words": [{"word": word, "probability": prob} for word, prob in zip(predicted_tokens, predicted_probs)] if predicted_tokens else []
    }

    user_data["interactions"].append(interaction)

    # Save back to JSON
    with open(user_file, "w") as f:
        json.dump(user_data, f, indent=4)




@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        first_name = request.form["first_name"].strip().capitalize()
        last_initial = request.form["last_initial"].strip().upper()
        grade = request.form["grade"].strip()
        q0= request.form.get("q0", "").strip()
        q1= request.form.get("q1", "").strip()
        q2= request.form.get("q2", "").strip()
        q3= request.form.get("q3", "").strip()


        user_id = str(uuid.uuid4())


        user_data = {
            "user_id": user_id,
            "first_name": first_name,
            "last_initial": last_initial,
            "grade": grade,
            "q0": q0,
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "interactions": []
        }


        user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
        with open(user_file, "w") as f:
            json.dump(user_data, f, indent=4)


        return redirect(url_for("home", user_id=user_id))

    return render_template("login.html")


# Home page after login get user ID
@app.route("/home")
def home():
    user_id = request.args.get("user_id")

    if not user_id:
        return redirect(url_for("login"))  # Redirect to login if no user_id

    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")

    if not os.path.exists(user_file):
        return redirect(url_for("login"))  # If user file is missing, force login

    with open(user_file, "r") as f:
        user_data = json.load(f)

    return render_template("index.html", user_data=user_data, user_id=user_id)


@app.route('/index', methods=['GET'])
def index():
    user_file = request.args.get('user_file', 'default_user.json')
    print("Received user file for ingredient interaction:", user_file)
    return render_template('index.html', user_file=user_file)



@app.route("/chat")
def chat():
    user_id = request.args.get("user_id")

    if not user_id:
        return redirect(url_for("login"))  # Redirect to login if no user_id

    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")

    if not os.path.exists(user_file):
        return redirect(url_for("login"))  # If user file is missing, force login

    return render_template('chat.html', user_id=user_id)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    model_name = request.form['model']
    user_id = request.form.get('user_id', 'unknown_user')
    temperature = float(request.form.get('temperature', 1.0))
    temp_slider = float(request.form.get('temp_slider', 1.0))
    p_value = float(request.form.get('p_value', 0.0))
    seed = int(request.form.get('seed', 42))
    top_k = 10

    if model_name in models:
        tokenizer = models[model_name]['tokenizer']
        model = models[model_name]['model']
        input_ids = tokenizer.encode(text, return_tensors='pt').tolist()[0]
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        predicted_tokens, probabilities, predicted_token_ids = get_next_word_predictions(model_name, text, top_k, temperature, p_value)
    else:
        return jsonify({'error': 'Model not supported'}), 400

    # **Log user interaction as "typed"**
    input_source = request.form.get("input_source", "typed")
    save_user_interaction(user_id, "typed", text, model_name, temperature, temp_slider, p_value, seed,
                          predicted_tokens, probabilities, selection_method=input_source )


    return jsonify({
        'input_text': text,
        'input_tokens': input_tokens,
        'input_token_ids': input_ids,
        'predicted_tokens': predicted_tokens,
        'predicted_token_ids': predicted_token_ids,
        'probabilities': probabilities,
        'temperature': temperature,
        'chart_temperature': temp_slider,
        'p_value': p_value,
        'seed': seed
    })


@app.route('/save_interaction', methods=['POST'])
def save_interaction():
    data = request.json
    user_id = data.get('user_id', 'unknown_user')
    interaction_type = data.get('interaction_type', 'unknown')  # "typed", "clicked", "auto-selected"
    selection_method = data.get('selection_method', interaction_type)  # Default to same as interaction_type
    input_text = data.get('input_text', '')
    selected_word = data.get('selected_word', None)
    model = data.get('model', 'unknown')
    temperature = float(data.get('temperature', 1.0))
    temp_slider = float(data.get('chart_temperature', 1.0))
    p_value = float(data.get('p_value', 0.0))
    seed = int(data.get('seed', 42))

    # Include predicted words and their probabilities
    predicted_tokens = data.get('predicted_tokens', [])  # List of suggested words
    predicted_probs = data.get('probabilities', [])  # Their probabilities

    save_user_interaction(user_id, interaction_type, input_text, model, temperature, temp_slider, p_value, seed,
                          predicted_tokens=predicted_tokens, predicted_probs=predicted_probs,
                          selected_word=selected_word, selection_method=selection_method)

    return jsonify({"status": "success"})




def clean_token(token):
    """Removes special BPE markers and ensures valid words"""
    token = token.lstrip('Ġ')  # Remove space marker
    token = token.strip()  # Remove extra spaces
    return token if token.isalnum() else None  # Keep only valid words

def get_next_word_predictions(model_name, text, top_k=10, temperature=1.0, p_value=0.0):
    tokenizer = models[model_name]['tokenizer']
    model = models[model_name]['model']
    input_ids = tokenizer.encode(text, return_tensors='pt')
    outputs = model(input_ids)
    predictions = outputs.logits[0, -1, :]

    scaled_predictions = predictions / temperature
    probs = torch.softmax(scaled_predictions, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)
    top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
    token_ids = top_indices.tolist()

    cleaned_tokens = []
    cleaned_probs = []
    cleaned_ids = []

    #Add words until their total probability reaches P-value threshold
    cumulative_prob = 0.0
    for token, prob, token_id in zip(top_tokens, top_probs.tolist(), token_ids):
        token = clean_token(token)
        if token:
            cleaned_tokens.append(token)
            cleaned_probs.append(prob)
            cleaned_ids.append(token_id)
            cumulative_prob += prob  # Sum up probabilities
            if cumulative_prob >= p_value:  # Stop adding once P-value is reached
                break

    # Ensure at least one token is always returned
    if not cleaned_tokens:
        cleaned_tokens.append(top_tokens[0])
        cleaned_probs.append(top_probs[0].item())
        cleaned_ids.append(top_indices[0].item())

    return cleaned_tokens, cleaned_probs, cleaned_ids


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_id = data.get('user_id', 'unknown_user')  # Get user_id from frontend
    user_input = data.get('query', '')
    age = data.get('age', '10')
    topic = data.get('topic', 'general knowledge')
    temperature = float(data.get('temperature', 0.7))
    top_p = float(data.get('top_p', 0.9))

    try:
        # Step 1: Get GPT-4 Response
        system_prompt = f"""
        You are a chatbot expert in {topic}. You are only allowed to answer questions related to {topic}.
        If the user asks a question outside of {topic}, you must respond with:
        "I can only answer questions about {topic}. Please ask something related to {topic}."

        Be concise, factual, and age-appropriate. Do not answer unsafe or inappropriate questions.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=80,
            temperature=temperature,
            top_p=top_p
        )

        message = response['choices'][0]['message']['content'].strip()
        input_tokens = user_input.split()
        output_tokens = message.split()

        # Step 2: Ask GPT-4 to Estimate Probabilities
        probability_prompt = f"Assign probability values (as percentages) to the words in this response:\n\n\"{message}\"\n\nFormat output as: word1 (prob1%), word2 (prob2%)..."
        prob_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an AI providing probability estimates for each word in a sentence."},
                      {"role": "user", "content": probability_prompt}],
            max_tokens=100
        )

        prob_message = prob_response['choices'][0]['message']['content'].strip()

        # Step 3: Parse GPT-4's Probability Response
        word_probabilities = []
        for token in output_tokens:
            found_prob = "N/A"
            for part in prob_message.split(","):
                if token in part:
                    try:
                        found_prob = float(part.split("(")[-1].replace("%)", ""))
                    except:
                        found_prob = "N/A"
            word_probabilities.append(found_prob)

        # Step 4: **Save user interaction to their JSON file**
        save_user_interaction(
            user_id=user_id,
            interaction_type="chat",
            input_text=user_input,
            model="gpt-4",
            temperature=temperature,
            temp_slider=0,  # Chat does not use temp_slider
            p_value=top_p,
            seed=0,  # Not applicable here
            predicted_tokens=output_tokens,
            predicted_probs=word_probabilities
        )

    except Exception as e:
        message = f"Error in API call: {str(e)}"
        output_tokens, word_probabilities = [], []

    return jsonify({
        'answer': message,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'word_probabilities': word_probabilities if word_probabilities else None,
        'total_tokens': len(input_tokens) + len(output_tokens)
    })



@app.route("/build_prompt")
def build_prompt():
    user_id = request.args.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if not os.path.exists(user_file):
        return redirect(url_for("login"))

    return render_template("build_prompt.html", user_id=user_id)  # make sure user_id is passed



@app.route('/generate_prompt_response', methods=['POST'])
def generate_prompt_response():
    data = request.json
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.7))
    p_value = float(data.get("p_value", 0.0))
    # seed = int(data.get("seed", 42))
    user_id = data.get("user_id")
    original_prompt = data.get("original_prompt")  # optional
    reflection = data.get("reflection_answer")     # optional

    try:
        # Generate story text with GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a friendly storyteller for children aged 7 to 13. Keep the story simple, fun, and imaginative. Use age-appropriate words."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=p_value,
            max_tokens=50
        )
        result = response['choices'][0]['message']['content'].strip()

        # Generate image from story text using DALL-E
        image_response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="256x256"
        )
        image_url = image_response['data'][0]['url']

        # Save interaction to user file
        if user_id:
            user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
            if os.path.exists(user_file):
                with open(user_file, "r") as f:
                    user_data = json.load(f)


                user_data.setdefault("prompt_history", []).append({
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "original_prompt": original_prompt,     # only present if user rewrote
                    "is_rewrite": bool(original_prompt),    # true/false
                    "temperature": temperature,
                    "p_value": p_value,
                    "response": result,
                    "image_url": image_url,
                    "reflection": reflection
                })


                with open(user_file, "w") as f:
                    json.dump(user_data, f, indent=4)

        return jsonify({"response": result, "image_url": image_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/save_custom_word', methods=['POST'])
def save_custom_word():
    data = request.json
    print("Received data:", data)  # <-- Debug print

    user_id = data.get("user_id")
    category = data.get("category")
    word = data.get("word")

    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    print(f"📄 Looking for file: {user_file}")

    if not os.path.exists(user_file):
        print(" User file not found!")
        return jsonify({"error": "User not found"}), 404

    with open(user_file, "r") as f:
        user_data = json.load(f)

    if "custom_words" not in user_data:
        user_data["custom_words"] = { "Characters": [], "Settings": [], "Actions": [] }

    if word not in user_data["custom_words"].get(category, []):
        user_data["custom_words"][category].append(word)

    with open(user_file, "w") as f:
        json.dump(user_data, f, indent=4)

    print(f"Word '{word}' saved under '{category}' for user {user_id}")
    return jsonify({"status": "success"})

@app.route('/download_user_data/<user_id>')
def download_user_data(user_id):
    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if os.path.exists(user_file):
        return send_file(user_file, as_attachment=True)
    return "User data not found", 404



@app.route("/save_questions", methods=["POST"])
def save_questions():
    data = request.get_json()
    user_id = data.get("user_id")
    q4 = data.get("q4", "").strip()
    q5 = data.get("q5", "").strip()
    q6 = data.get("q6", "").strip()

    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if not os.path.exists(user_file):
        return jsonify({"error": "User not found"}), 404

    # Load user file
    with open(user_file, "r") as f:
        user_data = json.load(f)

    # 💡 Only save if values are not empty
    if q4:
        user_data["q4"] = q4
    if q5:
        user_data["q5"] = q5
    if q6:
        user_data["q6"] = q6

    # Clean up bad keys (if any got in before)
    for key in ["q4", "q5", "q6"]:
        if key in user_data and user_data[key] == "":
            del user_data[key]

    with open(user_file, "w") as f:
        json.dump(user_data, f, indent=4)

    return jsonify({"status": "saved"})



@app.route("/save_reflection", methods=["POST"])
def save_reflection():
    data = request.get_json()
    user_id = data.get("user_id")
    reflection = data.get("reflection_answer", "").strip()

    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if not os.path.exists(user_file):
        return jsonify({"error": "User not found"}), 404

    with open(user_file, "r") as f:
        user_data = json.load(f)

    # Save to reflection_answers list
    if "reflection_answers" not in user_data:
        user_data["reflection_answers"] = []

    user_data["reflection_answers"].append({
        "timestamp": datetime.now().isoformat(),
        "answer": reflection
    })

    with open(user_file, "w") as f:
        json.dump(user_data, f, indent=4)

    return jsonify({"status": "saved"})


if __name__ == '__main__':
    app.run(debug=True)
