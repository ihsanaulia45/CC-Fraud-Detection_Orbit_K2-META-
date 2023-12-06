from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('RF_model.sav')  # Ganti dengan model Anda

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            if file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    predictions = model.predict(df)  # Ganti dengan logika prediksi Anda

                    # Hitung persentase 'Fraud' dan 'Not Fraud'
                    fraud_percentage = (predictions.sum() / len(predictions)) * 100
                    not_fraud_percentage = 100 - fraud_percentage

                    return render_template('result.html', fraud_percentage=fraud_percentage, not_fraud_percentage=not_fraud_percentage)
                except Exception as e:
                    return f"Terjadi kesalahan: {str(e)}"
            else:
                return "File harus berformat CSV."
        else:
            return "Tidak ada file yang diunggah."
    return "Metode tidak diizinkan."

if __name__ == '__main__':
    app.run(debug=True)
