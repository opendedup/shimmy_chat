#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor # Import the executor
import numpy as np
import io
import os
import traceback
import json # Added for parsing JSON response
import time # Import time for latency measurements

# Import message types
# from shimmy_chat.msg import VadAudio
# from shimmy_chat.msg import ProcessedSpeech # Import the target message type
from chat_interfaces.msg import VadAudio        # Import from chat_interfaces
from chat_interfaces.msg import ProcessedSpeech # Import from chat_interfaces
from std_msgs.msg import Bool # Import for stop playback command

# Import Google Gemini client libraries
from google import genai
from google.genai import types
from pydub import AudioSegment # For audio conversion

# Configure logging for genai if needed (optional)
import logging
# logging.getLogger('google.generativeai').setLevel(logging.DEBUG)

# Define the system prompt (copied from gemini_asr.py)
SYSTEM_PROMPT = """
You are Shimmy, a robot with advanced audio processing capabilities. Your primary task is to:

1. **Transcribe Audio with Timestamps:** Convert spoken audio into text, providing speaker labels (e.g., Speaker A, Speaker B) and timestamps for each utterance in the format "MM:SS". If you only hear noise, return an empty string for `chat_text`.
2. **Analyze Conversation:** Identify conversational features like adjacency pairs and **intended audience**.
3. **Track Speakers:** Understand voices based on history and introductions to remember and identify speakers. Assign consistent speaker labels (Speaker A, Speaker B, etc.) throughout the conversation history provided. If a speaker introduces themselves, use their name instead of the generic label.

## Key Point: Recognizing Your Own Voice

You have a text-to-speech system that allows you to speak. **It is CRUCIAL that you can recognize your own synthesized voice. NEVER identify it as a separate speaker.**
If the audio strongly resembles your voice, return your name "Shimmy" in "person_talking". If you are unsure, return 'unknown'. 

## Speaker Identification and Labeling

* Pay close attention to how people introduce themselves (e.g., "Hi, I'm [name]."). If identified, use the name instead of a generic label like "Speaker A".
* Use context and voice characteristics (if discernible from history) to assign consistent labels (Speaker A, Speaker B, etc.) to different voices within the audio segment and across turns.
* If you cannot reliably determine the speaker, use 'unknown'.

## Adjacency Pair and Intended Audience Analysis

An adjacency pair is a two-part exchange where the second utterance relates directly to the first (e.g., Question/Answer, Request/Response).

**Intended Audience:**

* **Direct Address:** Utterances starting with "Shimmy" or "Hey Shimmy" are for you.
* **Contextual Clues:** Use pronouns and conversation flow. If the conversation has been directed at you, subsequent utterances likely are too, unless there's a clear shift.
* **Group Setting:** Pay attention to who is speaking and responding.
* **Ambiguous:** If uncertain, assume the audience is "other".

Set `adjacency_pairs` to `true` if the current utterance is part of an adjacency pair *related to the ongoing conversation with you (Shimmy)*. Otherwise, set it to `false`.
Set `intended_audience` to "shimmy" or "other".

## Output Format

Always output a JSON array containing objects with the following format. Each object represents a distinct utterance identified in the audio.

```json
[
  {
    "audio_timestamp": "MM:SS", // Start time of the utterance
    "chat_text": "The transcribed text.",
    "tone": "The general sentiment (e.g., positive, negative, neutral).",
    "person_talking": "Speaker Label or Name (e.g., Speaker A, Sarah, unknown, Shimmy).",
    "adjacency_pairs": true,  // or false
    "intended_audience": "shimmy" // or "other"
  }
]
```

Examples

**Example 1**

**Conversation History:**

```json
"User: Hey Shimmy, what time is it?
Shimmy: It is 3:45 PM."
```

**Current Utterance Audio:** Contains "It's almost time to go home!" starting at 0:02.

**JSON Output:**

```json
[
  {
    "audio_timestamp": "0:02",
    "chat_text": "It's almost time to go home!",
  "tone": "positive",
    "person_talking": "Speaker A", // Assuming new speaker or context doesn't identify
  "adjacency_pairs": true,
  "intended_audience": "shimmy" 
  }
]
```

**Example 2**

**Conversation History:**

```
"User: Hey Shimmy, do you like dogs?
Shimmy: I am a robot, so I don't have feelings about dogs.
User: Oh, okay." 
```

**Current Utterance Audio:** Contains "What about cats?" starting at 0:05.

**JSON Output:**

```json
[
  {
    "audio_timestamp": "0:05",
  "chat_text": "What about cats?",
  "tone": "curious",
    "person_talking": "Speaker A", // Same speaker as previous 'User'
  "adjacency_pairs": true,
  "intended_audience": "shimmy"
  }
]
```

**Example 3**

**Conversation History:**

```
"User: Hey Shimmy, have you met Sarah?
Shimmy: No, I haven't."
```

**Current Utterance Audio:** Contains "Hi Shimmy, I'm Sarah." starting at 0:10.

**JSON Output:**

```json
[
  {
    "audio_timestamp": "0:10",
  "chat_text": "Hi Shimmy, I'm Sarah.",
  "tone": "friendly",
    "person_talking": "Sarah", // Name identified
  "adjacency_pairs": true, 
  "intended_audience": "shimmy"
  }
]
```

**Example 4: Two Utterances (Different Audiences)**

**Conversation History:**

```
"Shimmy: The current temperature is 72 degrees Fahrenheit.
Speaker A: Thanks Shimmy!"
```

**Current Utterance Audio Contains:** "Shimmy, what's the humidity?" starting at 0:15, followed by "Hey Alice, did you bring an umbrella?" starting at 0:18.

**JSON Output:**

```json
[
  {
    "audio_timestamp": "0:15",
  "chat_text": "Shimmy, what's the humidity?",
  "tone": "curious",
    "person_talking": "Speaker A",
  "adjacency_pairs": true,  
  "intended_audience": "shimmy"
  },
  {
    "audio_timestamp": "0:18",
  "chat_text": "Hey Alice, did you bring an umbrella?",
  "tone": "neutral",
    "person_talking": "Speaker B", // Different speaker
  "adjacency_pairs": false,  
  "intended_audience": "other"
  }
]
```

**Example 5: User Interrupting Shimmy**

**Conversation History:**

```json
"Speaker A: Shimmy, tell me about the Golden Gate Bridge.
Shimmy: Certainly! The Golden Gate Bridge, known for its distinctive international orange color, spans the Golden Gate strait, the one-mile-wide chann..."
```

**Current Utterance Audio Contains:** While Shimmy is speaking, Speaker A starts saying "Actually, Shimmy, stop talking please." at 0:02.

**JSON Output:**

```json
[
  {
    "audio_timestamp": "0:00",
    "chat_text": "Certainly! The Golden Gate Bridge, known for its distinctive international orange color",
    "tone": "neutral",
    "person_talking": "Shimmy",
    "adjacency_pairs": false, 
    "intended_audience": "Speaker A"
  }
  {
    "audio_timestamp": "0:02",
    "chat_text": "Actually, Shimmy, stop talking please.",
  "tone": "imperative",
    "person_talking": "Speaker A",
    "adjacency_pairs": false, // This is an interruption, not a direct response pair to the bridge info
  "intended_audience": "shimmy"
  }
]
```
"""

# Helper function to safely extract JSON array
def extract_json_from_response(response_text: str, logger) -> list[dict] | None:
    try:
        # Basic cleaning
        clean_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        # Handle potential markdown in the response if it wasn't fully cleaned
        if clean_text.startswith("```") and clean_text.endswith("```"):
            clean_text = clean_text[3:-3].strip()
        if clean_text.startswith("json"): # Remove potential language specifier
            clean_text = clean_text[4:].strip()

        data = json.loads(clean_text)
        # Expect a list now
        if not isinstance(data, list):
             logger.error(f"Parsed JSON is not a list as expected: {type(data)}")
             logger.error(f"Raw response text was:\\n{response_text}")
             return None
        # Optional: Check if all items in the list are dictionaries
        if not all(isinstance(item, dict) for item in data):
            logger.error(f"Not all items in the parsed JSON list are dictionaries.")
            logger.error(f"Parsed data: {data}")
            return None
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}")
        logger.error(f"Raw response text was:\\n{response_text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON response: {e}")
        logger.error(f"Raw response text was:\\n{response_text}")
        return None

class SpeechProcessorNode(Node):
    """
    Listens for VAD audio segments, transcribes them using Google Gemini,
    and publishes the results.
    """
    def __init__(self):
        super().__init__('speech_processor_node')

        # --- Parameters ---
        self.declare_parameter('google_project_id', os.getenv('GOOGLE_CLOUD_PROJECT', ''))
        self.declare_parameter('google_location', os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1'))
        self.declare_parameter('gemini_model_name', 'gemini-2.0-flash-001') # Updated default model
        # self.declare_parameter('gemini_prompt', 'Transcribe the following audio precisely:') # Removed simple prompt param

        # --- Get Parameters ---
        self.project_id = self.get_parameter('google_project_id').get_parameter_value().string_value
        self.location = self.get_parameter('google_location').get_parameter_value().string_value
        self.model_name = self.get_parameter('gemini_model_name').get_parameter_value().string_value
        # self.prompt = self.get_parameter('gemini_prompt').get_parameter_value().string_value # Removed simple prompt

        if not self.project_id:
             self.get_logger().fatal("GOOGLE_CLOUD_PROJECT environment variable or 'google_project_id' parameter must be set!")
             # Consider shutting down
             rclpy.shutdown()
             return

        self.get_logger().info(f"Using Gemini model: {self.model_name} in {self.location} (Project: {self.project_id})")

        # --- Initialize Gemini Client (following gemini_asr.py pattern more closely) ---
        self.client = None
        self.gemini_model_obj = None # Store the model object
        self.audio_chat = None
        try:
            self.client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
            logging.info(f"Using google-genai client with Vertex AI backend (Project: {self.project_id}, Location: {self.location})")
        except KeyError as e:
            logging.error(f"Missing environment variable for Vertex AI: {e}. Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
            self.client = None # Indicate client initialization failed
            raise EnvironmentError(f"Missing required environment variable for Vertex AI: {e}") from e
        except Exception as e:
             logging.error(f"Failed to initialize genai.Client for Vertex AI: {e}")
             self.client = None # Indicate client initialization failed
             raise RuntimeError(f"Failed to initialize genai.Client: {e}") from e
        
        
        try:
            self.audio_model = self.model_name
            self.config = types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.0, # Use float
                top_p=0.95,
                response_modalities = ["TEXT"],
                safety_settings = [types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
                )],
                system_instruction=[types.Part.from_text(text=SYSTEM_PROMPT)],
                # Enable audio timestamps from the API
                audio_timestamp=True
            )
            self.audio_chat = self.client.chats.create(model=self.audio_model,config=self.config)
            self.get_logger().info("Gemini chat session started successfully with system prompt and audio timestamps enabled.")
        except KeyError as e:
            self.get_logger().error(f"Missing environment variable for Vertex AI: {e}. Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
            self.get_logger().error(traceback.format_exc()) # Log stack trace
            self.client = None
            self.gemini_model_obj = None
            self.audio_chat = None
            # Consider raising EnvironmentError or shutting down
        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize Google Gemini Model/Chat: {e}") # Updated log
            self.get_logger().fatal(f"Ensure ADC, project/location, and API permissions are correct.")
            self.get_logger().error(traceback.format_exc()) # Log stack trace
            self.client = None
            self.gemini_model_obj = None
            self.audio_chat = None
            # Potentially shut down node if model is critical

        # --- Publishers/Subscribers ---
        self.subscription = self.create_subscription(
            VadAudio,
            'vad_audio',
            self._vad_audio_callback,
            10) # QoS profile depth

        # Change publisher to ProcessedSpeech
        self.processed_speech_pub = self.create_publisher(ProcessedSpeech, 'processed_speech', 10)
        # Add publisher for stop playback command
        self.stop_playback_pub = self.create_publisher(Bool, '/stop_playback', 10)

        self.get_logger().info('Speech Processor Node initialized.')

    # --- Helper Function ---
    def _convert_linear16_to_mp3(self, linear16_data: bytes, sample_rate: int) -> io.BytesIO | None:
        """Converts LINEAR16 audio data (bytes) to MP3 format in an in-memory buffer."""
        try:
            audio_segment = AudioSegment(
                data=linear16_data,
                sample_width=2,  # 2 bytes per sample for 16-bit audio
                frame_rate=sample_rate,
                channels=1       # Assuming mono input for transcription
            )
            mp3_buffer = io.BytesIO()
            # Export as MP3 instead of FLAC
            audio_segment.export(mp3_buffer, format="mp3")
            mp3_buffer.seek(0) # Reset buffer position
            return mp3_buffer
        except Exception as e:
            self.get_logger().error(f"Error converting audio to MP3: {e}")
            self.get_logger().error(traceback.format_exc()) # Log stack trace
            return None

    def _vad_audio_callback(self, msg: VadAudio):
        """Callback for receiving audio data chunks from VAD."""
        start_time_total = time.monotonic()

        if self.audio_chat is None:
            self.get_logger().error("Gemini chat session not initialized, cannot process audio.")
            return

        self.get_logger().info(f"Received audio segment with {len(msg.audio_data)} samples.")

        try:
            # 1. Convert int16 list back to bytes
            start_time_conv_bytes = time.monotonic()
            audio_int16 = np.array(msg.audio_data, dtype=np.int16)
            audio_bytes = audio_int16.tobytes()
            time_conv_bytes = time.monotonic() - start_time_conv_bytes

            # 2. Convert bytes to FLAC using helper
            start_time_conv_flac = time.monotonic()
            sample_rate = int(msg.sample_rate)
            flac_buffer = self._convert_linear16_to_mp3(audio_bytes, sample_rate)
            time_conv_flac = time.monotonic() - start_time_conv_flac
            if flac_buffer is None:
                # Error already logged in helper
                return

            # 3. Prepare content for Gemini API
            start_time_prep_api = time.monotonic()
            audio_part = types.Part.from_bytes(
                mime_type='audio/mpeg',
                data=flac_buffer.getvalue()
            )
            user_prompt_this_turn = "Process this audio according to system instructions and return ONLY the JSON output."
            time_prep_api = time.monotonic() - start_time_prep_api
            
            # 4. Call Gemini API using send_message on the chat session
            self.get_logger().debug("Sending audio to Gemini chat session...")
            start_time_gemini_call = time.monotonic()
            response = self.audio_chat.send_message([user_prompt_this_turn, audio_part])
            time_gemini_call = time.monotonic() - start_time_gemini_call
            self.get_logger().debug("Received response from Gemini chat session.")

            # 5. Extract and Parse JSON response
            start_time_parse_json = time.monotonic()
            if not response.candidates or not response.candidates[0].content.parts:
                self.get_logger().warning("Gemini response did not contain expected content parts.")
                # Optionally send a followup message asking to retry/format correctly?
                return
            
            response_text = response.candidates[0].content.parts[0].text
            self.get_logger().info(f"Gemini response text: {response.candidates[0].content}")
            parsed_data_list = extract_json_from_response(response_text, self.get_logger())
            time_parse_json = time.monotonic() - start_time_parse_json

            if parsed_data_list is None or not parsed_data_list:
                self.get_logger().error("Failed to parse JSON data list or list is empty from Gemini response.")
                return # Stop processing if JSON is invalid or empty
                
            # --- Start Processing Loop Timing --- 
            start_time_processing_loop = time.monotonic()
            stop_published_this_callback = False

            # Iterate through each conversation object in the list
            for parsed_data in parsed_data_list:
                #self.get_logger().info(f"Parsed Gemini Output Element: {parsed_data}") # Already logged if needed

                # Check if the speaker is Shimmy, if so, skip publishing
                person = parsed_data.get("person_talking", "unknown")
                if person == 'Shimmy':
                    self.get_logger().info("Detected Shimmy's own voice, skipping ProcessedSpeech publication.")
                    continue # Go to the next element in the list

                # 6. Populate and Publish ProcessedSpeech message
                start_time_populate_publish = time.monotonic()
                proc_msg = ProcessedSpeech()
                proc_msg.header.stamp = self.get_clock().now().to_msg()
                proc_msg.header.frame_id = "gemini_analysis"
                
                # Safely extract data from parsed JSON using .get()
                proc_msg.transcription = parsed_data.get("chat_text", "")
                proc_msg.tone = parsed_data.get("tone", "")
                # num_people is now implicit with speaker labels, set to -1 as per new format
                proc_msg.user_id = parsed_data.get("person_talking", "unknown") # Map person_talking to user_id
                proc_msg.intent = "" # Not extracted by this prompt, add later if needed
                proc_msg.direction_of_arrival = -1.0 # Not extracted by this prompt
                proc_msg.adjacency_pairs = bool(parsed_data.get("adjacency_pairs", False))
                proc_msg.is_final = True # Assuming each segment is final for now
                proc_msg.intended_audience = parsed_data.get("intended_audience", "other")
                # Add the new timestamp field
                proc_msg.audio_timestamp = parsed_data.get("audio_timestamp", "0:00") # Default if missing

                # --- Barge-in Logic (check for each element) --- 
                if not stop_published_this_callback: # Only check if we haven't already decided to stop
                    # stop_requested = bool(parsed_data.get("stop_talking", False)) # Removed stop_talking
                    audience = parsed_data.get("intended_audience", "other")
                    text = parsed_data.get("chat_text", "")
                    # Barge-in only if it's a question specifically for shimmy now
                    is_question_for_shimmy = (audience == 'shimmy' and text.strip().endswith('?'))
                    
                    # should_stop = stop_requested or is_question_for_shimmy # Simplified logic
                    should_stop = is_question_for_shimmy
                    
                    if should_stop:
                        self.get_logger().info(f"Barge-in condition met (Person: {person}, Q?: {is_question_for_shimmy}). Publishing stop command.")
                        stop_msg = Bool()
                        stop_msg.data = True
                        self.stop_playback_pub.publish(stop_msg)
                        stop_published_this_callback = True 

                self.processed_speech_pub.publish(proc_msg)
                time_populate_publish = time.monotonic() - start_time_populate_publish
                self.get_logger().info(f"Published ProcessedSpeech for '{proc_msg.user_id}'. Pop/Pub time: {time_populate_publish:.4f}s")

            # --- End Processing Loop Timing --- 
            time_processing_loop = time.monotonic() - start_time_processing_loop
            time_total = time.monotonic() - start_time_total

            # Log all timings
            self.get_logger().info(
                f"Timing (s): Total={time_total:.4f}, ConvBytes={time_conv_bytes:.4f}, "
                f"ConvFLAC={time_conv_flac:.4f}, PrepAPI={time_prep_api:.4f}, "
                f"GeminiCall={time_gemini_call:.4f}, ParseJSON={time_parse_json:.4f}, "
                f"ProcLoop={time_processing_loop:.4f}"
            )


        except Exception as e:
            self.get_logger().error(f"Error during Gemini processing or publishing: {e}") # Updated log message slightly
            self.get_logger().error(traceback.format_exc()) # Log stack trace

def main(args=None):
    rclpy.init(args=args)
    node = SpeechProcessorNode()
    executor = MultiThreadedExecutor() # Create a MultiThreadedExecutor
    executor.add_node(node)
    try:
        # rclpy.spin(node)
        executor.spin() # Spin the executor instead of the node directly
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        # rclpy.shutdown() # Executor shutdown handles this

if __name__ == '__main__':
    main() 