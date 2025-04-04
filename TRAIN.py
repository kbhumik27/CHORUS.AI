import numpy as np
import tensorflow as tf
import os
import json
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import datetime
import pickle
import matplotlib.pyplot as plt
import random  # Added to enable random shuffling

class NSynthDataLoader:
    def __init__(self, data_path: str, max_files: int = 1000):
        self.data_path = os.path.join(data_path, "nsynth-train")
        self.max_files = max_files
        self.metadata = {}
        self._load_metadata()
        
    def _load_metadata(self):
        metadata_path = os.path.join(self.data_path, "examples.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"Loaded metadata for {len(self.metadata)} samples")
    
    def extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        # Validate input audio
        if y.shape[0] == 0:
            raise ValueError("Empty audio input")
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        features = np.concatenate([mel_mean, mfccs_mean, contrast_mean])
        
        # Validate extracted features
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("Invalid features detected (NaN or Inf values)")
            
        return features

    def load_data(self):
        sample_data, audio_features = [], []
        audio_path = os.path.join(self.data_path, "audio")
        wav_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
        
        print(f"Loading {self.max_files} files from {len(wav_files)} total files")
        
        # Randomly shuffle the list of files so they are processed in random order
        random.shuffle(wav_files)
        
        processed_files = 0
        for wav_file in wav_files:
            if processed_files >= self.max_files:
                break
                
            try:
                file_id = wav_file.split('.')[0]
                if file_id not in self.metadata:
                    continue
                
                y, sr = librosa.load(os.path.join(audio_path, wav_file), sr=16000, duration=4.0)
                
                # Validate audio data
                if len(y) < sr * 0.5:  # Ensure at least 0.5 seconds of audio
                    print(f"Skipping {wav_file}: Too short")
                    continue
                    
                features = self.extract_features(y, sr)
                
                metadata = self.metadata[file_id]
                sample_params = [
                    float(metadata['pitch']) / 127.0,
                    float(metadata['velocity']) / 127.0,
                    float(metadata['instrument_family']) / 11.0,
                    float(metadata['instrument_source']) / 3.0
                ]
                
                sample_data.append(sample_params)
                audio_features.append(features)
                processed_files += 1
                
            except Exception as e:
                print(f"Error processing {wav_file}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(sample_data)} files")
        return np.array(sample_data), np.array(audio_features)

class NSynthSequencePreparation:
    def __init__(self, seq_length: int = 32):
        self.seq_length = seq_length
        self.scaler = StandardScaler()
        self.feature_dims = None
    
    def validate_scaler(self, data):
        """Validate scaler transformation"""
        transformed = self.scaler.transform(data)
        if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
            raise ValueError("Invalid scaling detected (NaN or Inf values)")
        return transformed

    def prepare_data(self, sample_data, audio_features):
        if len(sample_data) != len(audio_features):
            raise ValueError("Mismatched sample and feature lengths")
            
        combined_data = np.concatenate([sample_data, audio_features], axis=1)
        self.feature_dims = combined_data.shape[1]
        
        # Fit the scaler to the combined data before validating
        self.scaler.fit(combined_data)
        combined_data = self.validate_scaler(combined_data)
        
        sequences, targets = [], []
        for i in range(len(combined_data) - self.seq_length):
            sequences.append(combined_data[i:i + self.seq_length])
            targets.append(combined_data[i + self.seq_length])
        
        return np.array(sequences), np.array(targets)
    
    def save_scaler(self, path: str):
        """Save the fitted scaler and feature dimensions"""
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler has not been fitted to data")
            
        scaler_data = {
            'scaler': self.scaler,
            'feature_dims': self.feature_dims
        }
        
        with open(path, 'wb') as f:
            pickle.dump(scaler_data, f)
        print(f"Scaler saved to {path}")
    
    @staticmethod
    def load_scaler(path: str):
        """Load and validate saved scaler"""
        with open(path, 'rb') as f:
            scaler_data = pickle.load(f)
            
        if not isinstance(scaler_data, dict) or 'scaler' not in scaler_data:
            raise ValueError("Invalid scaler file format")
            
        return scaler_data['scaler'], scaler_data['feature_dims']

def create_model(input_shape, output_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # First LSTM layer
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(512, return_sequences=True)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    
    # Second LSTM layer
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(512)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    
    outputs = tf.keras.layers.Dense(output_shape, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def validate_model(model, val_data, val_targets):
    """Validate model predictions"""
    predictions = model.predict(val_data)
    mse = np.mean((predictions - val_targets) ** 2)
    print(f"Validation MSE: {mse}")
    return mse < 2.0  # Threshold based on your specific needs

def main():
    # Configuration
    data_path = "nsynth_small"
    max_files = 70000
    seq_length = 32
    batch_size = 64
    epochs = 10
    
    # Initialize components
    data_loader = NSynthDataLoader(data_path, max_files=max_files)
    sequence_prep = NSynthSequencePreparation(seq_length=seq_length)
    
    # Load and prepare data
    print("Loading data...")
    sample_data, audio_features = data_loader.load_data()
    print("Preparing sequences...")
    x, y = sequence_prep.prepare_data(sample_data, audio_features)
    
    # Split data
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    # Save scaler
    sequence_prep.save_scaler('scaler.pkl')
    
    # Create and compile model
    print("Creating model...")
    model = create_model(
        input_shape=(x.shape[1], x.shape[2]), 
        output_shape=y.shape[1]
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Validate final model
    print("Validating model...")
    if validate_model(model, x_val, y_val):
        print("Model validation successful")
        model.save('nsynth_model.h5')
        
        # Save training history
        with open('training_historylstm.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
    else:
        print("Model validation failed")

if __name__ == "__main__":
    main()
