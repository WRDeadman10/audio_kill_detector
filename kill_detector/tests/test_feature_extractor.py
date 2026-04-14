import unittest
import numpy as np
import librosa
import os
from kill_detector.src.feature_extractor import extract_features

class TestFeatureExtractor(unittest.TestCase):
    """
    Tests for the feature_extractor function.
    """

    def setUp(self):
        # Create a temporary directory and dummy audio file for testing
        self.temp_dir = "temp_audio_test"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.dummy_audio_path = os.path.join(self.temp_dir, "dummy_audio.wav")

        # Create a standard synthetic audio array (e.g., 1 second of silence)
        self.sr = 22050
        self.duration = 1.0
        self.audio_data = np.zeros(int(self.sr * self.duration)).astype(np.float32)
        
        # Save the dummy audio file
        librosa.output.write(self.dummy_audio_path, self.audio_data, sr=self.sr)

    def tearDown(self):
        # Clean up the temporary directory and files
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_deterministic_fixed_size_vector(self):
        """
        Tests that the feature extractor returns a fixed-size vector 
        for a standard audio clip and that the result is deterministic.
        """
        # Run extraction once
        features1 = extract_features(self.dummy_audio_path)
        
        # Run extraction again to check determinism
        features2 = extract_features(self.dummy_audio_path)
        
        # Check if the shapes are identical and the values are close (deterministic)
        self.assertEqual(features1.shape, features2.shape)
        np.testing.assert_allclose(features1, features2, atol=1e-5)

        # Check if the output is a numpy array
        self.assertTrue(isinstance(features1, np.ndarray))
        
        # Check if the size is non-zero
        self.assertGreater(features1.size, 0, "Feature vector should not be empty.")

    def test_short_audio_array_safety(self):
        """
        Tests that the feature extractor handles very short audio arrays safely 
        without crashing or producing NaNs/Infs.
        """
        # Create an extremely short audio array (e.g., 10 samples)
        short_duration = 0.1
        short_audio_data = np.random.rand(int(self.sr * short_duration)).astype(np.float32)
        short_audio_path = os.path.join(self.temp_dir, "short_audio.wav")
        librosa.output.write(short_audio_path, short_audio_data, sr=self.sr)

        try:
            features = extract_features(short_audio_path)
            
            # Check if the output is a numpy array
            self.assertTrue(isinstance(features, np.ndarray))
            
            # Check for NaNs or Infs
            self.assertFalse(np.isnan(features).any(), "Feature vector contains NaN values.")
            self.assertFalse(np.isinf(features).any(), "Feature vector contains Inf values.")
            
        except Exception as e:
            self.fail(f"extract_features failed on short audio array: {e}")

if __name__ == '__main__':
    unittest.main()
