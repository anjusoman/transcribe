from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import wave
import threading
import uuid
import os

basepath = os.path.join("audio")
recording_queue = []

condition = threading.Condition()

def initialize_model(model_size):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return model

def add_to_queue(audio_filepath):
    recording_queue.append(audio_filepath)

def transcribe_audio(model):
    global recording_queue
    t = threading.current_thread()
    t.alive = True
    while t.alive:
        with condition:
            while len(recording_queue) == 0 and not stop_recording:
                condition.wait()

            if stop_recording:  # Check again after waking up
                break

            audio_filepath = recording_queue.pop(0)

        print("Transcribing...")

        segments, info = model.transcribe(audio_filepath, beam_size=5)

        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

        os.remove(audio_filepath)

def record_audio(duration=1, sample_rate=44100, channels=1, dtype='float64'):
    global recording_queue
    t = threading.current_thread()
    t.alive = True
    while t.alive:
        filepath = os.path.join(basepath, str(uuid.uuid4()))
        wave_filepath = os.path.normpath(filepath + ".wav")

        print("Recording...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype=dtype)
        sd.wait()  # Wait until the recording is finished

        # Save the audio to a WAV file
        with wave.open(wave_filepath, 'wb') as wf:
            wf.setnchannels(channels)  # Set channels
            wf.setsampwidth(2)  # 2 bytes per sample
            wf.setframerate(sample_rate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())  # Convert to int16

        if os.path.exists(wave_filepath):
            with condition:
                recording_queue.append(wave_filepath)
                condition.notify_all()  # Notify all waiting threads
        else:
            print(f"Failed to create {wave_filepath}")

def main():
    global recording_queue, stop_recording
    
    try:
        # Create model
        print("Loading transcribe model... ")
        model = WhisperModel("faster-distil-whisper-small.en")
        #model = WhisperModel("large-v2")
        print("Model loaded successfully.")

    
        # Create threads
        record_thread = threading.Thread(target=record_audio)
        transcribe_thread = threading.Thread(target=transcribe_audio, args=(model,))

        # Start the threads
        record_thread.start()
        transcribe_thread.start()

        # If the child thread is still running
        while record_thread.is_alive() or transcribe_thread.is_alive():
		    # Try to join the child thread back to parent for 0.5 seconds so interrupt can be processed
            if record_thread.is_alive(): record_thread.join(0.1)
            if transcribe_thread.is_alive(): transcribe_thread.join(0.1)

        # Wait for the threads to complete
        record_thread.join()
        transcribe_thread.join()

    except KeyboardInterrupt as e:
        # Set the flag to stop recording
        record_thread.alive = False
        transcribe_thread.alive = False

        # Join the threads again to ensure clean exit
        if record_thread is not None and record_thread.is_alive():
            record_thread.join()

        if transcribe_thread is not None and transcribe_thread.is_alive():
            transcribe_thread.join()

        # Delete extra files
        for file in recording_queue:
            os.remove(file)

        print("Stopped recording and transcription.")
    
    except Exception as e:
        # Handle the exception
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
