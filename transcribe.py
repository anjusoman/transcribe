from faster_whisper import WhisperModel
import threading
import numpy as np
import heapq
import pyaudio
import os
import webrtcvad
import numpy as np

basepath = os.path.join("audio")
recording_queue = []
printing_queue = []
transcription = []

num_transcribe_threads = 8

condition = threading.Condition()
print_condition = threading.Condition()

def initialize_model(model_size):
    return WhisperModel(model_size, device="cpu", compute_type="int8")

def add_to_queue(audio_filepath):
    with condition:
        heapq.heappush(recording_queue, audio_filepath)
        condition.notify_all()

def print_segments():
    global printing_queue
    t = threading.current_thread()
    t.alive = True

    next_priority = 0
    while t.alive:
        with print_condition:
            # Wait until an item with the target priority is in the queue
            while t.alive and (len(printing_queue) == 0 or printing_queue[0][0] != next_priority):
                print_condition.wait()

            if not t.alive:  # Exit if thread is no longer alive
                break

            # Pop and print the item with the correct priority
            priority, segments = heapq.heappop(printing_queue)
            for segment in segments:
                #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                print(segment.text)
                transcription.append(segment.text)

            next_priority += 1

def transcribe_audio(model):
    global recording_queue, printing_queue
    t = threading.current_thread()
    t.alive = True
    while t.alive:
        with condition:
            while t.alive and len(recording_queue) == 0:
                condition.wait()

            if not t.alive:
                break

            rank, audio_df = recording_queue.pop(0)

        segments, _ = model.transcribe(audio_df, beam_size=5)

        with print_condition:
            heapq.heappush(printing_queue, (rank, segments))
            print_condition.notify_all()  # Notify print_segments to check the queue

def record_audio(duration=2.5, sample_rate=44100, channels=1, dtype='float64'):
    global recording_queue
    t = threading.current_thread()
    t.alive = True

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    buffer = bytes()
    rank = 0

    while t.alive:
        data = stream.read(4096, exception_on_overflow=False)
        buffer += data

        if len(buffer) > 16000 * 2 * 5:
            audio_data = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0
            if has_voice(audio_data):
                with condition:
                    recording_queue.append((rank, audio_data))
                    rank += 1
                    condition.notify_all()  # Notify all transcribe threads
            buffer = bytes()  # Reset buffer for next batch


def has_voice(audio_data, sample_rate=16000, frame_duration=30, aggressiveness=2):

    # Initialize webrtcvad with the specified aggressiveness level
    vad = webrtcvad.Vad(aggressiveness)
    
    # Calculate the number of bytes per frame
    bytes_per_frame = int(sample_rate * frame_duration / 1000) * 2  # 16-bit audio
    
    # Process audio in chunks of the specified frame duration
    for i in range(0, len(audio_data), bytes_per_frame):
        frame = audio_data[i:i + bytes_per_frame]
        
        # Check if the frame contains voice activity
        if len(frame) == bytes_per_frame and vad.is_speech(frame, sample_rate):
            return True  # Voice detected in this frame
    
    return False  # No voice detected in any frames


def main():
    global recording_queue
    
    try:
        # Load model
        print("Loading transcribe model... ")
        model = "faster-whisper-small"
        model = WhisperModel(model)
        print("Model loaded successfully.")

        # Start threads
        record_thread = threading.Thread(target=record_audio)
        record_thread.start()

        transcribe_threads = []
        for _ in range(num_transcribe_threads):
            transcribe_thread = threading.Thread(target=transcribe_audio, args=(model,))
            transcribe_thread.start()
            transcribe_threads.append(transcribe_thread)

        print_thread = threading.Thread(target=print_segments)
        print_thread.start()

        # Join threads safely
        while record_thread.is_alive() or any(t.is_alive() for t in transcribe_threads):
            record_thread.join(0.1)
            print_thread.join(0.1)
            for transcribe_thread in transcribe_threads:
                transcribe_thread.join(0.1)

    except KeyboardInterrupt:
        record_thread.alive = False
        print_thread.alive = False
        for transcribe_thread in transcribe_threads:
            transcribe_thread.alive = False

        with condition:
            condition.notify_all()  # Wake up threads waiting on condition
        with print_condition:
            print_condition.notify_all()  # Wake up threads waiting on print_condition

        # Final join
        record_thread.join()
        print_thread.join()
        for transcribe_thread in transcribe_threads:
            transcribe_thread.join()

        with open("transcription.txt", 'w') as f:
            f.writelines([s + ' ' for s in transcription])

        print("Stopped recording, transcription, and printing.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
