import json
import os
import logging
import time
import random
import multiprocessing
import pyaudio
from playsound import playsound
from flask import Flask
from flask_sock import Sock
from dotenv import load_dotenv
from ibm_watson import TextToSpeechV1
from ibm_watson.websocket import SynthesizeCallback
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model

# Set up logging (default to DEBUG) 
logging.basicConfig(level=logging.DEBUG)

# Run with gunicorn:
# gunicorn -b :5000 --workers 4  --threads 5 wscommentary:app
app = Flask(__name__)
sock = Sock(app)

# load .env
load_dotenv()

global model_id
model_id = os.getenv("GENAI_MODEL")

default_model_parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 200,
    "temperature": 0.2,
    "top_k": 50,
    "top_p": 1,
    "repetition_penalty": 1.05,
    "stop_sequences": ["}"]
}

global player_profile_model_parameters
player_profile_model_parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 250,
    "temperature": 0.4,
    "top_k": 50,
    "top_p": 1,
    "repetition_penalty": 1.05,
    "stop_sequences": ["}"]
}


global no_processing_required_types       
no_processing_required_types = ["ping","shot_playback_done","selected_club","game_and_environment_data"]

# Final commentary file

project_id = os.getenv("GENAI_PROJECT_ID")
global iam_api_key
iam_api_key = os.getenv("API_KEY", None)
global wml_api_endpoint 
wml_api_endpoint = os.getenv("GENAI_URL", None)
global wml_creds
wml_creds = {
    "url": os.getenv("GENAI_URL", None),
    "apikey": iam_api_key
}

# Instantiate a model proxy object to send your requests
global single_threaded_model
single_threaded_model = Model(
    model_id=model_id,
    params=default_model_parameters,
    credentials=wml_creds,
    project_id=project_id
    )

global iam_authenticator
iam_authenticator = IAMAuthenticator(iam_api_key)
global single_threaded_tts_service
single_threaded_tts_service = TextToSpeechV1(authenticator=iam_authenticator)
single_threaded_tts_service.set_service_url(os.getenv("TTS_URL"))

player_profile_prompt_prefix = """
You are a golf commentator known for your golf knowledge. You are introducing a golf player as they are about to hit a shot at the 7th hole of the Pebble Beach Golf Links course . You will be given an input JSON containing information about the golf player. Use this information to output 4 full sentences of summary about the player. Do not use a player name. Use a formal personality with a good-natured sense of humor. Output only the summary commentary in the following JSON structure: {"commentary":"Generated commentary goes here"}
Input:

"""

end_commentary_prompt_template="""
You are a golf commentator known for your golf knowledge. You are providing commentary about a shot that has just been hit. You will be given an input containing information about the shot results. Use this information to output 3 full sentences describing the shot's results. Do not use a player name. Assume the distance to pin is in feet. A Final Terrain Type of "water" or "bunker" is considered a bad shot and a hazard. A Final Terrain Type of "green" is considered a good shot. All other Final Terrain Types are considered average shots. Use a formal personality with a good-natured sense of humor. Output only the summary commentary in the following JSON structure: {"commentary":"Generated commentary goes here"}

Input:
Shot Number: 1
Par: 3
Final Terrain Type: {terrain_type}
Distance to pin: {pin_distance}
Shot Shape: {shot_shape}

JSON:

"""
final_commentary_file = ""

# Delete everything in a string right after  the last occurrence og a given char
def delete_after_last_char(string, char):
    last_index = string.rfind(char)  # Find the last index of the character
    if last_index != -1:  # If the character is found
        return string[:last_index + 1]  # Return the string up to and including the last occurrence of the character
    else:
        return string  # If the character is not found, return the original string

# Add SSML to  text to synthesize
def enhance_with_SSML(text)->str:
    # For now just avoiding run ons where you expect a break 
   
    local_text = text.replace(' - ','<break strength="medium"/>')
    local_text = local_text.replace('...','<break strength="medium"/>')

    return local_text



# TO DO replace with training TTS instance directly
def alternative_pronunciations(tts_input):
    fixed_input = tts_input.replace('putting',"<phoneme alphabet='ibm' ph='.1pH.0diG' />")   
    fixed_input = fixed_input.replace('Putting',"<phoneme alphabet='ibm' ph='.1pH.0diG' />")
    fixed_input = fixed_input.replace(' lead',"<phoneme alphabet='ibm' ph='.1lid' />")
    fixed_input = fixed_input.replace('Lead',"<phoneme alphabet='ibm' ph='.1lid' />")

    return fixed_input

# Simplify shot data by retaining only the fields we are using 
def get_shot_profile(payload_data):
    shot_profile = {}
    shot_profile['shot_shape'] = payload_data['shot_complete']['data']['shot_shape']
    shot_profile['shot_time'] = payload_data['shot_complete']['data']['segments'][-1]['points'][-1]['time']
    final_segment = payload_data['shot_complete']['data']['snapshots'][-1]
    shot_profile['terrain_type'] = final_segment['terrain_type']
    shot_profile['pin_distance'] = final_segment['pin_distance']
   

    return shot_profile

# Returns init commentary file based on shot profile
def get_init_commentary_file(shot_profile):
    random_file_number = str(random.randint(1, 7))

    if shot_profile['terrain_type'] == "green":
        return "audio/good_" + random_file_number + ".mp3"
    elif shot_profile['terrain_type'] == "water" or shot_profile['terrain_type'] == "tee_box" or shot_profile['terrain_type'] == "bunker":
        return "audio/bad_" + random_file_number + ".mp3"
    
    return "audio/average_" + random_file_number + ".mp3"


#  Wrapper to play audio in a blocking mode
class Play(object):
    """
    Wrapper to play the audio in a blocking mode
    """
    def __init__(self):
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 22050
        self.chunk = 1024
        self.pyaudio = None
        self.stream = None

    def start_streaming(self):
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self._open_stream()
        self._start_stream()

    def _open_stream(self):
        stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            output=True,
            frames_per_buffer=self.chunk,
            start=False
        )
        return stream

    def _start_stream(self):
        self.stream.start_stream()

    def write_stream(self, audio_stream):
        self.stream.write(audio_stream)

    def complete_playing(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

# Callback for TTS websocket  that streams  synthesized sound to the speakers
class LiveSynthesizeCallback(SynthesizeCallback):
    def __init__(self):
        SynthesizeCallback.__init__(self)
        self.play = Play()

    def on_connected(self):
        stop = time.perf_counter()
        logging.debug(f"Opening stream to play. Elapsed time is {stop-start}")  
        logging.debug(f"Stop is {stop}, start is {start}")  
        self.play.start_streaming()

    def on_error(self, error):
        logging.error('Error received: {}'.format(error))

    def on_timing_information(self, timing_information):
        logging.debug(timing_information)

    def on_audio_stream(self, audio_stream):
        self.play.write_stream(audio_stream)

    def on_close(self):
        logging.debug('Completed synthesizing')
        self.play.complete_playing()

tts_callback_live = LiveSynthesizeCallback()

# Callback for TTS websocket  that writes synthesized sound to a file 
class FileSynthesizeCallback(SynthesizeCallback):
    def __init__(self, player_id):
        SynthesizeCallback.__init__(self)
        logging.debug(f"FileSynthesizeCallback instance writingto file audio/{player_id}.wav")
        self.wav = open("audio/" + player_id + ".wav","wb")

    def on_connected(self):
        stop = time.perf_counter()

    def on_error(self, error):
        logging.error('Error received: {}'.format(error))

    def on_timing_information(self, timing_information):
        logging.debug(timing_information)

    def on_audio_stream(self, audio_stream):
        self.wav.write(audio_stream)

    def on_close(self):
        logging.debug(f"FileSynthesizeCallback completed synthesizing to file {self.wav.name}")
        self.wav.close()

# Run asynchronously to generate the player commentary audio and save in a file
def generate_player_commentary(player_id, player_profile):
  multi_threaded_tts_callback_file = FileSynthesizeCallback(player_id)
  multi_threaded_tts_service = TextToSpeechV1(authenticator=iam_authenticator) 
  multi_threaded_tts_service.set_service_url(os.getenv("TTS_PLAYER_PROFILE_URL"))

  model = Model(
    model_id=model_id,
    params=player_profile_model_parameters,
    credentials=wml_creds,
    project_id=project_id
  )

  # Remove unwanted keys before sending to LLM 
  player_profile.pop('player_id', None)
  player_profile.pop('ballsLostPerRound', None)
  player_profile.pop('displayName', None)
  player_profile.pop('familyName', None)

  # Send to LLM to get text commentary
  prompt = player_profile_prompt_prefix + json.dumps(player_profile) + '\n' +  "JSON:\n"
  llm_response = model.generate_text(prompt)
  logging.debug("*** Start LLM response  ***")
  logging.debug(f"LLM response = {llm_response}")
  logging.debug("*** Start LLM response  ***")
  response_dict = json.loads(delete_after_last_char(llm_response, '}'))

  logging.debug("Synthesizing player commentary ...")
  ssml_enhanced = enhance_with_SSML(response_dict['commentary'])
  logging.debug(f"SSML enhanced commentary = {ssml_enhanced}")
  local_start = time.perf_counter()
  multi_threaded_tts_service.synthesize_using_websocket(ssml_enhanced,  
                                                        multi_threaded_tts_callback_file,                                 
                                                        accept='audio/wav',
                                                        voice="en-US_EmmaExpressive")
  local_stop = time.perf_counter()
  logging.debug(f"Synthesizing player commentary took {local_stop-local_start} seconds")
  return

@sock.route('/watsonx')
def watsonx(ws):
    while True:
      payload_raw = ws.receive()
      payload_data = json.loads(payload_raw)

  
      if payload_data["type"] in no_processing_required_types:
          # Handle requests that require no processing 
          logging.debug(f"Handling ws message type {payload_data['type']}")
          ws.send(f"{payload_data['type']} response")
          continue
      
      if payload_data["type"] == "user_profile":
         # Player profile received
         # TO DO implement asynchronous generation of player profile
         logging.debug(f"Handling ws message type {payload_data['type']}")
         thread = multiprocessing.Process(target=generate_player_commentary, args=(payload_data['user_profile']['player_id'], payload_data['user_profile']))
         thread.start()
         ws.send(f"Player commentary generating for player_id {payload_data['user_profile']['player_id']}")
         continue
         
      if payload_data["type"] == "user_data":
         # Player login received
         logging.debug(f"Handling ws message type {payload_data['type']}")
         # Play user commentary 10 seconds after receiving this 
         player_commentary_audio_file = 'audio/' + payload_data["user_profile"]["data"]["player_id"] + '.wav'
         logging.debug(f"Waiting {os.getenv('COMMENTARY_START_DELAY','10')} seconds before playing player commentary")
         time.sleep(int(os.getenv("COMMENTARY_START_DELAY","10")))
         playsound(player_commentary_audio_file, block=True)   

      elif payload_data["type"] == "shot_data": 
        logging.debug(f"Handling ws message type {payload_data['type']}")
        shot_profile = get_shot_profile(payload_data)
        init_commmentary_file = get_init_commentary_file(shot_profile)
     
        logging.debug(f"playing clip {init_commmentary_file}.wav")
        playsound(init_commmentary_file, block=False)

        logging.debug("Starting timer")
        start = time.perf_counter()
       
       
        prompt = end_commentary_prompt_template.format(shot_shape=shot_profile['shot_shape'], player_name=player_name,
                                                       terrain_type=shot_profile['terrain_type'], pin_distance=str(shot_profile['pin_distance']/30.48) + ' ft')
        logging.debug("*** Start prompt ***")
        logging.debug(prompt)
        logging.debug("*** End prompt ***")
        llm_response = single_threaded_model.generate_text(prompt)
        response_dict = json.loads(delete_after_last_char(llm_response, '}'))
       
        logging.debug("*** Start LLM response  ***")
        logging.debug(llm_response)
        logging.debug("*** End LLM response ***")

        # Check if OK to start final commentary 
        time_to_shot_complete  = shot_profile['shot_time'] - (time.perf_counter() - start)

        while time_to_shot_complete  > 1.5:
            logging.debug(f"Shot complete in {time_to_shot_complete} secs - too early to start end commentary ")
            time.sleep(1)
            time_to_shot_complete  = shot_profile['shot_time'] - (time.perf_counter() - start)

        logging.debug(f"Starting end commentary with {time_to_shot_complete} secs before shot complete")                   
        single_threaded_tts_service.synthesize_using_websocket(alternative_pronunciations(response_dict["commentary"]),
                                                               tts_callback_live,
                                                               accept='audio/wav',
                                                               voice="en-US_EmmaExpressive")
                                            
                                
      ws.send('Msg processed')



if __name__ == '__main__':
    app.run(threaded=False, processes=6)


