import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import unittest

class TestBeatmapProcessing(unittest.TestCase):
    def test_feature_extraction(self):
        """
        Tests whether the feature extraction from beatmaps is accurate.
        """
        # Implement test cases for preprocessing and feature extraction
        # [test cases continue...]
        pass

def read_beatmap(file_path):
    """ Reads and parses an Osu! beatmap file. """
    beatmap_data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            section = None
            for line in file:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    section = line[1:-1]
                    beatmap_data[section] = []
                elif section and line:
                    beatmap_data[section].append(line)
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
    return beatmap_data

def preprocess_data(beatmaps):
    """ Extracts and normalizes features from a list of beatmaps. """
    inputs, targets = [], []
    for beatmap in beatmaps:
        if 'HitObjects' in beatmap:
            hit_objects = beatmap['HitObjects']
            sequence = [parse_hit_object(obj) for obj in hit_objects]
            sequence = np.array(sequence)
            if len(sequence) > 10:
                inputs.extend([sequence[i:i+10] for i in range(len(sequence) - 10)])
                targets.extend([sequence[i+10] for i in range(len(sequence) - 10)])
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)

def parse_hit_object(obj):
    """ Parses hit objects and extracts features including type and duration when applicable. """
    parts = obj.split(',')
    x, y, time, type_info = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    features = [x, y, time, 0, 0, 0]  # Circle by default
    if type_info & 2:
        features[3] = 1  # Slider
    elif type_info & 8:
        features[4] = 1  # Spinner
    return features

def build_model(input_shape, num_classes, learning_rate=0.0001):
    """ Builds and compiles an LSTM model for sequence generation. """
    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))
    return model

def train_model(model, inputs, targets):
    """ Trains the LSTM model. """
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2)
    model.fit(inputs_train, targets_train, epochs=10, batch_size=64)
    return model, inputs_test, targets_test

def evaluate_model(model, inputs_test, targets_test):
    """ Evaluates the model using MSE as the metric. """
    predictions = model.predict(inputs_test)
    mse = MeanSquaredError()
    mse.update_state(targets_test, predictions)
    return mse.result().numpy()

def plot_hit_objects_interactive(data):
    """ Creates an interactive visualization of hit objects. """
    fig = go.Figure()
    for sequence in data:
        for obj in sequence:
            x, y, type_info = obj[0], obj[1], obj[3:]
            obj_type = 'slider' if type_info[0] == 1 else 'spinner' if type_info[1] == 1 else 'circle'
            fig.add_trace(go.Scattergl(x=[x], y=[y], mode='markers', name=obj_type.capitalize()))
    fig.update_layout(title='Interactive Hit Objects Visualization')
    fig.show()

def generate_beatmap(model, seed, constraints):
    """ Generate a new beatmap based on the seed and user constraints. """
    generated_sequence = seed
    current_input = seed
    for _ in range(100):  # Assuming generation of 100 steps
        next_step = model.predict(current_input[np.newaxis, :, :])[-1]
        # Apply constraints to modify next_step before appending
        current_input = np.append(current_input, [next_step], axis=0)[1:]  # Slide window
        generated_sequence = np.append(generated_sequence, [next_step], axis=0)
    return generated_sequence


def evaluate_beatmap(beatmap):
    """Evaluate the generated beatmap both subjectively and objectively."""
    evaluation_results = {
        'difficulty': calculate_difficulty(beatmap),
        'balance': check_balance(beatmap),
        'player_feedback': get_player_feedback(beatmap)  # Hypothetical function
    }
    return evaluation_results

def calculate_difficulty(beatmap):
    # Example function to calculate difficulty
    return np.mean([x[2] for x in beatmap])  # Simplified difficulty calculation

def check_balance(beatmap):
    # Example function to check for a good balance of elements
    types = [x[3] for x in beatmap]
    return types.count(1) / len(types)  # Proportion of sliders, for example

def get_player_feedback(beatmap):
    # Collect feedback from players, which could be simulated or real
    return "Mostly positive"  # Placeholder for actual feedback mechanism


if __name__ == '__main__':
    beatmap_folder = 'path_to_beatmaps'
    beatmaps = [read_beatmap(os.path.join(beatmap_folder, file)) for file in os.listdir(beatmap_folder) if file.endswith('.osu')]
    inputs, targets = preprocess_data(beatmaps)
    model = build_model((10, 6), 3)  # Example parameters
    trained_model, inputs_test, targets_test = train_model(model, inputs, targets)
    mse_score = evaluate_model(trained_model, inputs_test, targets_test)
    print(f"Model MSE: {mse_score}")
    plot_hit_objects_interactive(inputs[:100])  # Example subset


