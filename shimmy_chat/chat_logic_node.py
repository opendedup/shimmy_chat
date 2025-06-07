#!/usr/bin/env python3

import logging
import rclpy
from rclpy.node import Node
import os
import traceback

# --- Gemini/Vertex AI Imports ---
from google import genai
from google.genai import types
# ------------------------------

# Import message types
# from std_msgs.msg import String # No longer subscribing to simple string
# from shimmy_chat.msg import ProcessedSpeech # Subscribe to richer message
from chat_interfaces.msg import ProcessedSpeech # Import from chat_interfaces
from std_msgs.msg import String # Still publish String for TTS node

class ChatLogicNode(Node):
    """
    Subscribes to processed speech results, determines an appropriate response
    if addressed to Shimmy using Gemini, and publishes the text to be spoken.
    """
    def __init__(self):
        super().__init__('chat_logic_node')

        # --- Parameters ---
        self.declare_parameter('robot_name', 'shimmy') # Name to check in intended_audience
        # REMOVED: self.declare_parameter('response_prefix', 'Okay, you said: ')
        # --- Gemini Parameters ---
        self.declare_parameter('project_id', '') # Optional: Defaults to GOOGLE_CLOUD_PROJECT env var
        self.declare_parameter('location', 'us-central1')
        self.declare_parameter('gemini_model_name', 'gemini-2.0-flash-001')
        self.declare_parameter('chat_system_prompt', 'You are a helpful robot named Shimmy.')
        self.declare_parameter('gemini_temperature', 1.0)
        self.declare_parameter('gemini_top_p', 0.95)
        self.declare_parameter('gemini_max_output_tokens', 8192)
        # ------------------------

        # --- Get Parameters --- 
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value.lower()
        # REMOVED: self.response_prefix = self.get_parameter('response_prefix').get_parameter_value().string_value
        # --- Get Gemini Parameters ---
        self.project_id = self.get_parameter('project_id').get_parameter_value().string_value or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.location = self.get_parameter('location').get_parameter_value().string_value
        self.gemini_model_name = self.get_parameter('gemini_model_name').get_parameter_value().string_value
        self.chat_system_prompt = self.get_parameter('chat_system_prompt').get_parameter_value().string_value
        self.gemini_temperature = self.get_parameter('gemini_temperature').get_parameter_value().double_value
        self.gemini_top_p = self.get_parameter('gemini_top_p').get_parameter_value().double_value
        self.gemini_max_output_tokens = self.get_parameter('gemini_max_output_tokens').get_parameter_value().integer_value
        # ---------------------------

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
            self.model = self.gemini_model_name
            self.config = types.GenerateContentConfig(
                max_output_tokens=self.gemini_max_output_tokens,
                temperature=self.gemini_temperature, # Use float
                top_p=self.gemini_top_p,
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
                system_instruction=[types.Part.from_text(text=self.chat_system_prompt)],
            )
            self.chat = self.client.chats.create(model=self.model,config=self.config)
            self.get_logger().info("Gemini chat session started successfully with system prompt.")
        except KeyError as e:
            self.get_logger().error(f"Missing environment variable for Vertex AI: {e}. Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
            self.get_logger().error(traceback.format_exc()) # Log stack trace
            self.client = None
            self.chat = None
            # Consider raising EnvironmentError or shutting down
        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize Google Gemini Model/Chat: {e}") # Updated log
            self.get_logger().fatal(f"Ensure ADC, project/location, and API permissions are correct.")
            self.get_logger().error(traceback.format_exc()) # Log stack trace
            self.client = None
            self.chat = None
            # Potentially shut down node if model is critical

        # --- Publishers/Subscribers ---
        self.subscription = self.create_subscription(
            ProcessedSpeech,
            'processed_speech',
            self._processed_speech_callback,
            10)

        self.text_to_speak_pub = self.create_publisher(String, 'text_to_speak', 10)

        self.get_logger().info('Chat Logic Node initialized.')

    def _processed_speech_callback(self, msg: ProcessedSpeech):
        """Callback for receiving processed speech results. Generates response using Gemini."""
        self.get_logger().info(
            f"Received processed speech: "
            f"User='{msg.user_id}', Audience='{msg.intended_audience}', "
            f"Text='{msg.transcription[:50]}...'"
        )

        # --- Response Logic with Gemini --- 
        response_text = "" # Default to no response

        if self.chat is None:
             self.get_logger().error("Gemini chat not initialized. Cannot generate response.")
             return

        # Check if the message is intended for the robot (case-insensitive)
        if msg.intended_audience.lower() == self.robot_name:
            if msg.transcription: # Only generate response if there's text
                user_input = f"{msg.user_id or 'User'} said: {msg.transcription}"
                self.get_logger().info(f"Sending to Gemini: '{user_input}'")
                try:
                    stream = self.chat.send_message_stream(user_input)

                    chunk_buffer = "" # Initialize buffer for accumulating text chunks
                    sentence_endings = ('.', '?', '!') # Define sentence endings

                    for chunk in stream:
                        if chunk.candidates and chunk.candidates[0].content.parts:
                            for part in chunk.candidates[0].content.parts:
                                if hasattr(part, 'text') and part.text.strip():
                                    chunk_buffer += part.text # Append text to buffer

                                    # Check if buffer ends with a sentence marker and has > 10 words
                                    stripped_buffer = chunk_buffer.strip()
                                    if stripped_buffer.endswith(sentence_endings):
                                        words = stripped_buffer.split()
                                        if len(words) > 10:
                                            # Publish if buffer ends with sentence and has > 10 words
                                            publish_text = stripped_buffer
                                            self.get_logger().debug(f"Publishing sentence chunk (> 10 words): '{publish_text}'")
                                            pub_msg = String()
                                            pub_msg.data = publish_text
                                            self.text_to_speak_pub.publish(pub_msg)
                                            chunk_buffer = "" # Reset buffer

                    # After the stream finishes, publish any remaining text in the buffer
                    if chunk_buffer.strip():
                        publish_text = chunk_buffer.strip()
                        self.get_logger().debug(f"Publishing remaining text: '{publish_text}'")
                        pub_msg = String()
                        pub_msg.data = publish_text
                        self.text_to_speak_pub.publish(pub_msg)

                except Exception as e:
                    self.get_logger().error(f"Error calling Gemini API stream: {e}")
                    self.get_logger().error(traceback.format_exc())
                    # Optional: Send a default error message
                    # response_text = "Sorry, I encountered an error."

            else:
                 self.get_logger().info("Message intended for Shimmy, but transcription is empty. No response.")
        else:
            self.get_logger().info(f"Message not intended for {self.robot_name}, ignoring.")
            return

def main(args=None):
    rclpy.init(args=args)
    node = ChatLogicNode()
    if node.client is None:
        rclpy.logging.get_logger("chat_logic_node_main").fatal(
            "Gemini client failed to initialize. Please check credentials, project ID, and network connection. Shutting down."
        )
        return # Exit if Gemini setup failed
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 