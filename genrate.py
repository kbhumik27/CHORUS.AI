import numpy as np
import tensorflow as tf
import librosa
import pickle
import pretty_midi
from sklearn.preprocessing import StandardScaler

class NSynthGenerator:
    def __init__(self, model_path: str, scaler_path: str, seq_length: int = 32):
        """Load trained model and scaler."""
        self.model = tf.keras.models.load_model(model_path)
        self.model.compile(optimizer="adam", loss="mse")  # Explicitly compile the model
        self.seq_length = seq_length
        self.scaler = self._load_scaler(scaler_path)

    def _load_scaler(self, scaler_path):
        """Load the scaler from a pickle file."""
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            if isinstance(scaler_data, dict) and 'scaler' in scaler_data:
                return scaler_data['scaler']
            else:
                raise TypeError(f"Expected dictionary with 'scaler', but got {type(scaler_data)}")

    def create_seed_sequence(self, 
                             pitch: float = 60,  # Middle C
                             velocity: float = 100,
                             instrument_family: int = 1,  # Piano
                             instrument_source: int = 2,  # Acoustic
                             random_init: bool = True):
        """Create an initial sequence for music generation."""
        # Normalize parameters as during training
        params = np.array([
            pitch / 127.0,
            velocity / 127.0,
            instrument_family / 11.0,
            instrument_source / 3.0
        ])
        # Generate audio features or use zeros based on random_init
        if random_init:
            audio_features = np.random.normal(0, 1, 148)  # e.g. 148 audio features
        else:
            audio_features = np.zeros(148)
        initial_vector = np.concatenate([params, audio_features])
        
        if self.scaler is None:
            raise ValueError("Scaler was not properly loaded.")
        
        initial_vector = self.scaler.transform(initial_vector.reshape(1, -1))
        # Repeat the vector to form the seed sequence
        seed_sequence = np.tile(initial_vector, (self.seq_length, 1))
        return seed_sequence

    def generate_sequence(self, seed_sequence, num_steps: int = 1000, noise_level: float = 0.01):
        """Generate a new sequence using the trained model.
        
        Args:
            seed_sequence: Initial sequence (numpy array) of shape (seq_length, feature_dim)
            num_steps: Number of additional time steps to generate
            noise_level: Standard deviation of noise added to the prediction (to reduce distortion)
        """
        generated_sequence = seed_sequence.copy()
        for _ in range(num_steps):
            # Use the last seq_length frames as input
            input_sequence = np.expand_dims(generated_sequence[-self.seq_length:], axis=0)
            next_vector = self.model.predict(input_sequence, verbose=0)
            # Add slight noise to the prediction
            noise = np.random.normal(0, noise_level, next_vector.shape)
            next_vector += noise
            # Clip to a reasonable range
            next_vector = np.clip(next_vector, -1.5, 1.5)
            generated_sequence = np.vstack([generated_sequence, next_vector])
        return generated_sequence

def sequence_to_midi(sequence, output_midi_path, time_per_step=0.5):
    """
    Convert the generated sequence into a MIDI file.
    
    Assumptions:
      - Column 0: Normalized pitch (pitch/127.0)
      - Column 1: Normalized velocity (velocity/127.0)
    
    Each row in the sequence is mapped to a note event with a fixed duration.
    
    Args:
        sequence: Generated feature sequence (numpy array) of shape (num_steps, feature_dim)
        output_midi_path: Path where the MIDI file will be saved
        time_per_step: Duration (in seconds) of each generated step (note length)
    """
    notes = []
    for i, vec in enumerate(sequence):
        # Denormalize pitch and velocity
        norm_pitch = vec[0]
        norm_velocity = vec[1]
        pitch = int(np.clip(round(norm_pitch * 127), 0, 127))
        velocity = int(np.clip(round(norm_velocity * 127), 0, 127))
        start_time = i * time_per_step
        end_time = start_time + time_per_step
        notes.append((pitch, velocity, start_time, end_time))
    
    # Create a new MIDI object and instrument (e.g., Acoustic Grand Piano)
    midi_obj = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for pitch, velocity, start_time, end_time in notes:
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
        instrument.notes.append(note)
    midi_obj.instruments.append(instrument)
    midi_obj.write(output_midi_path)

def generate_midi(model_path: str, scaler_path: str, output_midi_path: str,
                  pitch: float = 60, velocity: float = 100,
                  instrument_family: int = 0, instrument_source: int = 0,
                  num_steps: int = 100, time_per_step: float = 0.5):
    """
    Main function to generate a MIDI file from the trained NSynth model.
    
    Args:
        model_path: Path to the trained model file (e.g., "nsynth_model.h5")
        scaler_path: Path to the scaler pickle file (e.g., "scaler.pkl")
        output_midi_path: Where the MIDI file will be saved
        pitch, velocity, instrument_family, instrument_source: Seed parameters
        num_steps: Number of steps (time frames) to generate after the seed sequence
        time_per_step: Duration in seconds of each generated time step (determines note length)
    """
    generator = NSynthGenerator(model_path, scaler_path)
    seed_sequence = generator.create_seed_sequence(
        pitch=pitch,
        velocity=velocity,
        instrument_family=instrument_family,
        instrument_source=instrument_source
    )
    generated_sequence = generator.generate_sequence(seed_sequence, num_steps=num_steps)
    sequence_to_midi(generated_sequence, output_midi_path, time_per_step=time_per_step)
    return generated_sequence

if __name__ == "__main__":
    model_path = "nsynth_model.h5"
    scaler_path = "scaler.pkl"
    output_midi_path = "generated_music3.mid"
    
    try:
        # Adjust num_steps and time_per_step to achieve your desired musical duration
        sequence = generate_midi(
            model_path=model_path,
            scaler_path=scaler_path,
            output_midi_path=output_midi_path,
            pitch=100,
            velocity=170,
            instrument_family=11,
            instrument_source=5,
            num_steps=200,      # Number of generated frames (modify as needed)
            time_per_step=0.25  # Duration per frame in seconds (modify as needed)
        )
        print("MIDI generation completed! MIDI saved to:", output_midi_path)
    except Exception as e:
        print(f"Error: {e}")
