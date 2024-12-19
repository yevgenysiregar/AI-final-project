from flask import Flask, render_template, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import io
import base64
from datetime import timedelta
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Non-interactive backend for server environments

app = Flask(__name__)

# Load scaler and model
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "transformer_with_global_attention.keras"

scaler = joblib.load(SCALER_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# Load dataset
DATASET_PATH = "normalized_data.csv"
df = pd.read_csv(DATASET_PATH, parse_dates=["Datetime"])

# Add time-based features
df['Hour'] = df['Datetime'].dt.hour
df['Minute'] = df['Datetime'].dt.minute
df['Day_of_Week'] = df['Datetime'].dt.dayofweek
df['Day'] = df['Datetime'].dt.day

# Define features used for training
required_features = [
    "Global_active_power", "Global_intensity", "Sub_metering_1",
    "Sub_metering_2", "Sub_metering_3", "Hour", "Day_of_Week", "Day", "Minute"
]

# Extract past data in 30-minute intervals
window_size = 30
past_data = df.iloc[-(window_size * 2)::30]  # Take every 30th row for 30-minute intervals

def generate_plot(x, y, prediction=None, future_time=None, title="Graph", xlabel="Time", ylabel="Global Active Power"):
    """
    Generate a plot for past data and a single predicted point.
    """
    plt.figure(figsize=(8, 4))

    # Plot past data in blue
    plt.plot(x, y, marker='o', linestyle='-', color='blue', label="Past Data")

    # Red connection and prediction point
    if prediction is not None and future_time is not None:
        plt.plot([x[-1], future_time], [y[-1], prediction], linestyle='-', color='red', label="Prediction Connection")
        plt.plot(future_time, prediction, 'ro', label="Predicted Point")

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/past_graph', methods=['GET'])
def past_graph():
    """
    Generate the graph for past 'Global_active_power' values sampled every 30 minutes.
    """
    # Subset past data using window_size = 30
    sampled_data = df.iloc[-window_size * 30::30]  # Take every 30th row for 30-min interval
    time_values = sampled_data['Datetime'].dt.strftime('%H:%M')  # Format x-axis to HH:MM
    power_values = sampled_data['Global_active_power']
    
    # Generate and return the graph
    graph_url = generate_plot(
        x=time_values, 
        y=power_values, 
        title="Past Data (30-Minute Intervals)", 
        xlabel="Time"
    )
    return jsonify({"graph": graph_url})


@app.route('/future_graph', methods=['POST'])
def future_graph():
    """
    Predict one step (30 minutes ahead) and generate a graph connecting past and future data.
    """
    try:
        # Sample past data for 30-minute intervals
        sampled_data = df.iloc[-window_size * 30::30]  # Take every 30th row to get 30-minute intervals
        time_values_past = sampled_data['Datetime']
        power_values_past = sampled_data['Global_active_power']

        # Prepare input features and scale the data
        features = sampled_data[required_features].iloc[-window_size:]
        scaled_features = scaler.transform(features.to_numpy())

        # Reshape for model input and make prediction
        input_sequence = scaled_features.reshape(1, window_size, len(required_features))
        prediction_scaled = model.predict(input_sequence).flatten()[0]

        # Inverse scale the prediction to the original value
        unscaled_prediction = scaler.inverse_transform(
            [[prediction_scaled] + [0] * (len(required_features) - 1)]
        )[0][0]  # Only take 'Global_active_power'

        # Prepare future timestamp
        last_time = time_values_past.iloc[-1]
        future_time = last_time + timedelta(minutes=30)

        # Combine past data and predicted point
        time_values_combined = pd.concat(
            [time_values_past, pd.Series([future_time])], ignore_index=True
        )
        power_values_combined = pd.concat(
            [power_values_past, pd.Series([unscaled_prediction])], ignore_index=True
        )

        # Generate graph with proper x-axis intervals
        plt.figure(figsize=(6, 4))
        plt.plot(time_values_combined[:-1], power_values_combined[:-1], 
                 marker='o', linestyle='-', color='blue', label="Past Data")
        plt.plot(time_values_combined[-2:], power_values_combined[-2:], 
                 marker='o', linestyle='-', color='red', label="Prediction (30 Minutes Ahead)")

        # Format the x-axis to show only 30-minute time intervals
        plt.xticks(time_values_combined[::1], time_values_combined.dt.strftime('%H:%M'), rotation=45)

        plt.title("Past Data with Future Prediction (30-Minute Intervals)")
        plt.ylabel("Global Active Power")
        plt.xlabel("Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Convert graph to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return jsonify({"graph": f"data:image/png;base64,{graph_url}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
