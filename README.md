# Setup:

1. Create a Python Virtual Environment

        python3 -m venv venv

2. Enter the Python Virtual Environment

        source venv/bin/activate

3. Install the neccessary libraries

        pip install -r requirements.txt

4. Set up the faster whisper model

        ct2-transformers-converter --model openai/whisper-small --output_dir faster-whisper-small --copy_files tokenizer.json --quantization float16


# Running the program

1. Enter the Python Virtual Environment

        source venv/bin/activate

2. Run the transcribe program

        python3 transcribe.py



