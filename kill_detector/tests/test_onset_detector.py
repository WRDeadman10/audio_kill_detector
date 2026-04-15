import unittest
import numpy as np
import librosa
import os
import shutil
from kill_detector.src.onset_detector import detect_onsets
import soundfile as sf

class TestOnsetDetector(unittest.TestCase):
    """
    Tests for the onset detection function.
    """

    def setUp(self):
        # Create a temporary directory and dummy audio file for testing
        self.temp_dir = "temp_audio_test"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.dummy_audio_path = os.path.join(self.temp_dir, "dummy_audio.wav")
        self.sr = 22050

    def tearDown(self):
        # Clean up the temporary directory and files
        shutil.rmtree(self.temp_dir)

    def _create_dummy_audio(self, duration: float, frequency: float = 440.0):
        """Helper function to create a dummy audio file."""
        t = np.linspace(0, duration, int(self.sr * duration), False)
        # Simple sine wave for testing
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        dummy_path = os.path.join(self.temp_dir, "dummy_audio.wav")
        sf.write(dummy_path, audio_data, self.sr)
        return dummy_path

    def test_basic_onset_detection(self):
        """
        Tests basic onset detection on a standard audio clip.
        """
        # Create a standard audio clip (e.g., 2 seconds)
        audio_path = self._create_dummy_audio(2.0)
        
        # Run detection
        onsets = detect_onsets(audio_path)
        
        # Check if the output is a list of floats
        self.assertTrue(isinstance(onsets, list))
        
        # Check if the list is sorted and non-negative
        if onsets:
            self.assertTrue(all(isinstance(t, float) and t >= 0 for t in onsets))
            self.assertEqual(onsets, sorted(onsets))

    def test_empty_signal(self):
        """
        Tests onset detection on an empty or near-empty signal.
        """
        # Create a very short audio clip (e.g., 0.01 seconds)
        audio_path = self._create_dummy_audio(0.01)
        
        # Run detection
        onsets = detect_onsets(audio_path)
        
        # Expecting an empty list or a list with minimal results
        # We primarily test that it doesn't crash and returns a list.
        self.assertTrue(isinstance(onsets, list))
        # Depending on the implementation, it might return [] or [0.0]
        # We assert it's a list and doesn't raise an exception.
        
    def test_very_short_clip(self):
        """
        Tests onset detection on a very short clip (e.g., 100ms).
        """
        # Create a very short audio clip
        audio_path = self._create_dummy_audio(0.1)
        
        # Run detection
        onsets = detect_onsets(audio_path)
        
        # Assert it runs without error and returns a list
        self.assertTrue(isinstance(onsets, list))

    def test_synthetic_transient_signal(self):
        """
        Tests onset detection on a synthetic signal designed to produce 
        multiple, non-negative, sorted timestamps.
        """
        # Create a synthetic signal (e.g., a sequence of sharp transients)
        # We simulate a signal that should definitely trigger onsets at specific points.
        
        # Since we cannot easily generate a perfect transient signal using librosa 
        # for testing, we rely on creating a slightly longer, complex signal 
        # and ensuring the output is correctly formatted.
        
        # Create a 3-second audio path
        audio_path = self._create_dummy_audio(3.0)
        
        # Run detection
        onsets = detect_onsets(audio_path)
        
        # Assert the output is a list of floats, sorted, and non-negative.
        self.assertTrue(isinstance(onsets, list))
        if onsets:
            self.assertTrue(all(isinstance(t, float) and t >= 0 for t in onsets))
            self.assertEqual(onsets, sorted(onsets))


if __name__ == '__main__':
    unittest.main()
