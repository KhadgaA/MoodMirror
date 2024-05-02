
This program implements a real-time speaker emotion recognition system using a pre-trained Long Short-Term Memory (LSTM) model. The program is designed to take live audio input from the microphone, process the audio data, and predict the corresponding emotion.

Flow of the Program:

1. Load the pre-trained LSTM model:
   - The program loads the pre-trained LSTM model from a JSON file (saved_model_path) and its corresponding weights (saved_weights_path).
   - The model architecture and weights are loaded using the Keras library.
   - The model is compiled with the same parameters as the original training process.

2. Preprocess the audio input:
   - The preprocess() function is responsible for preprocessing the audio input before feeding it to the model.
   - It converts the raw audio data to an AudioSegment object using the pydub library.
   - The audio is then normalized, trimmed for silence, and padded to a fixed length.
   - Noise reduction is applied using the noisereduce library.
   - Features (RMS, ZCR, and MFCCs) are extracted from the audio data using the librosa library.
   - The extracted features are concatenated and reshaped to match the input format expected by the LSTM model.

3. Emotion recognition:
   - The EmotionRecogniser() function handles the real-time emotion recognition process.
   - It receives the audio stream and the new audio chunk from the microphone.
   - The new audio chunk is preprocessed using the preprocess() function.
   - The preprocessed data is fed to the pre-trained LSTM model for prediction.
   - The model outputs probabilities for each emotion class.
   - The emotion with the highest probability is considered the predicted emotion.
   - The program maintains a stream of audio data by concatenating the new chunk with the previous stream.
   - The predicted emotion and its probability distribution are returned.

4. Streaming audio input:
   - The program utilizes the gradio library to create a user interface with a microphone input.
   - The gradio.Interface() function is used to set up the user interface, where the EmotionRecogniser() function is called with the streaming audio input.
   - The interface allows the user to provide live audio input from the microphone.
   - As the user speaks, the program continuously processes the audio stream and updates the predicted emotion and its probability distribution in real-time.

To run the program, simply execute the Python script. The program will start listening to the microphone input, and you can speak into the microphone to see the real-time emotion recognition results displayed in the user interface.

Note: The program assumes that the necessary dependencies (e.g., Keras, TensorFlow, librosa, pydub, noisereduce, gradio) are installed. Additionally, the pre-trained LSTM model and its weights should be available in the specified file paths (saved_model_path and saved_weights_path).
