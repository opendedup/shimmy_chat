#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

# Import message types
from std_msgs.msg import String
# from shimmy_chat.msg import VadAudio # Use this to publish audio compatible with AudioInterfaceNode
from chat_interfaces.msg import VadAudio # Import from chat_interfaces

# Import Google Cloud Text-to-Speech client library
from google.cloud import texttospeech
import os # To potentially check for credentials env variable

class SpeechSynthesisNode(Node):
    """
    Subscribes to text messages, synthesizes speech using Google Cloud TTS,
    and publishes the resulting audio data.
    """
    def __init__(self):
        super().__init__('speech_synthesis_node')

        # --- Parameters ---
        # Authentication (implicitly uses ADC - Application Default Credentials)
        # Ensure ADC are set up (e.g., `gcloud auth application-default login` or GOOGLE_APPLICATION_CREDENTIALS env var)
        self.declare_parameter('tts_language_code', 'en-US')
        self.declare_parameter('tts_voice_name', 'en-US-Standard-J') # Example voice
        self.declare_parameter('tts_speaking_rate', 1.0) # Default speaking rate
        self.declare_parameter('tts_pitch', 0.0) # Default pitch
        self.declare_parameter('tts_sample_rate', 16000) # Must match AudioInterfaceNode!

        # --- Get Parameters ---
        self.language_code = self.get_parameter('tts_language_code').get_parameter_value().string_value
        self.voice_name = self.get_parameter('tts_voice_name').get_parameter_value().string_value
        self.speaking_rate = self.get_parameter('tts_speaking_rate').get_parameter_value().double_value
        self.pitch = self.get_parameter('tts_pitch').get_parameter_value().double_value
        self.sample_rate_hertz = self.get_parameter('tts_sample_rate').get_parameter_value().integer_value

        self.get_logger().info(f"Using TTS Voice: {self.voice_name} ({self.language_code})")
        self.get_logger().info(f"TTS Config: Rate={self.speaking_rate}, Pitch={self.pitch}, SampleRate={self.sample_rate_hertz}Hz")

        # --- Initialize TTS Client ---
        try:
            self.tts_client = texttospeech.TextToSpeechClient()
            self.get_logger().info("Google Cloud TTS client initialized successfully.")
            # Check if credentials seem available (basic check)
            if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("GOOGLE_CLOUD_PROJECT"):
                 self.get_logger().warning("GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_PROJECT env var not set. Ensure ADC is configured.")
        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize Google Cloud TTS client: {e}")
            self.get_logger().fatal("Ensure Application Default Credentials (ADC) are configured.")
            self.tts_client = None
            # Potentially shut down node if TTS is critical

        # --- Publishers/Subscribers ---
        self.subscription = self.create_subscription(
            String,
            'text_to_speak',
            self._text_to_speak_callback,
            10) # QoS profile depth
        
        self.playback_audio_pub = self.create_publisher(VadAudio, 'playback_audio', 10)
        
        self.get_logger().info('Speech Synthesis Node initialized.')

    def _text_to_speak_callback(self, msg: String):
        """Callback for receiving text to be synthesized."""
        if self.tts_client is None:
            self.get_logger().error("TTS client not available, cannot synthesize speech.")
            return
            
        text_to_synthesize = msg.data
        if not text_to_synthesize:
            self.get_logger().warn("Received empty string for synthesis, ignoring.")
            return
            
        self.get_logger().info(f"Synthesizing speech for: '{text_to_synthesize}'")

        try:
            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)

            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                name=self.voice_name
            )

            # Select the type of audio file you want returned (LINEAR16 for PCM)
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate_hertz,
                speaking_rate=self.speaking_rate,
                pitch=self.pitch
            )

            # Perform the text-to-speech request
            response = self.tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # --- Process Response --- 
            audio_bytes = response.audio_content
            self.get_logger().info(f"Received {len(audio_bytes)} bytes of synthesized audio.")

            if not audio_bytes:
                 self.get_logger().warning("TTS returned empty audio content.")
                 return

            # Convert audio bytes (int16) to list of ints for VadAudio message
            audio_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_list_int16 = audio_np_int16.tolist()

            # Publish as VadAudio message
            pub_msg = VadAudio()
            pub_msg.header.stamp = self.get_clock().now().to_msg()
            pub_msg.header.frame_id = "tts_output"
            pub_msg.audio_data = audio_list_int16
            pub_msg.sample_rate = float(self.sample_rate_hertz)
            
            self.playback_audio_pub.publish(pub_msg)
            self.get_logger().info(f"Published synthesized audio for playback of size {len(audio_list_int16)}.")

        except Exception as e:
            self.get_logger().error(f"Error during speech synthesis or publishing: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SpeechSynthesisNode()
    try:
        if node.tts_client is None:
             node.get_logger().warning("TTS Client failed to initialize. Node will run but cannot synthesize speech.")
             # Decide if node should exit here based on requirements
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 