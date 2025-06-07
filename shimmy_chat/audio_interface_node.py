#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pasimple as pa
import pulsectl
import numpy as np
import collections
import time
import queue
import threading
import contextlib
import sys
import wave
from datetime import datetime
import os

# Import WebRTC APM components
from webrtc_audio_processing import AudioProcessingModule, AudioProcessingStreamConfig, NSConfig, VADConfig # AECConfig might be needed for fine-tuning, but basic enable works

# Import custom/standard message types
from chat_interfaces.msg import VadAudio
from std_msgs.msg import Bool # For playback state and stop command

# No longer using whisper_trt.vad or sounddevice

class AudioInterfaceNode(Node):
    def __init__(self):
        super().__init__('audio_interface_node')

        # --- Parameters ---
        self.declare_parameter('input_device', '')
        self.declare_parameter('output_device', '')
        self.declare_parameter('sample_rate', 16000) # Unified sample rate for input, output, and APM
        self.declare_parameter('output_volume', 1.0)
        self.declare_parameter('channels', 1) # APM works best with mono, forcing this for simplicity

        # VAD logic parameters (retained, but now based on 10ms APM frames)
        self.declare_parameter('vad_silence_ms', 500)
        self.declare_parameter('vad_padding_ms', 200)

        # WebRTC APM specific parameters
        self.declare_parameter('apm_aec_suppression_level', 0) # 0: moderate, 1: high, 2: aggressive for AEC3. Some bindings might have different interpretations.
        self.declare_parameter('apm_aecm_enable', False) # Use AEC Mobile (simpler, less CPU)
        self.declare_parameter('apm_ns_level', 2)      # 0:low, 1:moderate, 2:high, 3:very high
        self.declare_parameter('apm_vad_likelihood', 1) # 0:aggressive ... 3:conservative for VAD quality

        self.declare_parameter('debug_save_vad_wav', False)
        self.declare_parameter('debug_wav_output_dir', '/tmp/vad_audio')

        # --- Get Parameters ---
        self.input_device_param = self.get_parameter('input_device').get_parameter_value().string_value
        self.output_device_param = self.get_parameter('output_device').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.output_volume = max(0.0, min(1.0, self.get_parameter('output_volume').get_parameter_value().double_value))
        _channels_param = self.get_parameter('channels').get_parameter_value().integer_value
        if _channels_param != 1:
            self.get_logger().warning(f"Parameter 'channels' was {_channels_param}, but forcing to 1 for WebRTC APM compatibility.")
        self.channels = 1 # Force mono for APM

        self.vad_silence_ms = self.get_parameter('vad_silence_ms').get_parameter_value().integer_value
        self.vad_padding_ms = self.get_parameter('vad_padding_ms').get_parameter_value().integer_value

        self.apm_aec_suppression_level = self.get_parameter('apm_aec_suppression_level').get_parameter_value().integer_value
        self.apm_aecm_enable = self.get_parameter('apm_aecm_enable').get_parameter_value().bool_value
        self.apm_ns_level = self.get_parameter('apm_ns_level').get_parameter_value().integer_value
        self.apm_vad_likelihood = self.get_parameter('apm_vad_likelihood').get_parameter_value().integer_value

        self.debug_save_vad_wav = self.get_parameter('debug_save_vad_wav').get_parameter_value().bool_value
        self.debug_wav_output_dir = self.get_parameter('debug_wav_output_dir').get_parameter_value().string_value

        self.get_logger().info(f"Unified Sample Rate: {self.sample_rate} Hz, Channels: {self.channels}")
        self.get_logger().info(f"VAD Logic: Silence={self.vad_silence_ms}ms, Padding={self.vad_padding_ms}ms")
        self.get_logger().info(f"APM Params: AEC Suppress={self.apm_aec_suppression_level}, AECM={self.apm_aecm_enable}, NS Level={self.apm_ns_level}, VAD Likelihood={self.apm_vad_likelihood}")

        if self.debug_save_vad_wav:
            self.get_logger().info(f"Debugging enabled: Saving VAD WAV files to '{self.debug_wav_output_dir}'")
            try:
                os.makedirs(self.debug_wav_output_dir, exist_ok=True)
            except OSError as e:
                 self.get_logger().error(f"Could not create WAV output directory '{self.debug_wav_output_dir}': {e}. Disabling WAV saving.")
                 self.debug_save_vad_wav = False

        # --- WebRTC APM Setup ---
        self.apm_frame_duration_ms = 10 # APM processes in 10ms frames
        self.apm_samples_per_frame = int(self.sample_rate * self.apm_frame_duration_ms / 1000)
        self.apm_bytes_per_sample = 2 # For 16-bit audio
        self.apm_bytes_per_frame = self.apm_samples_per_frame * self.apm_bytes_per_sample * self.channels

        try:
            apm_stream_config = AudioProcessingStreamConfig(sample_rate_hz=self.sample_rate, num_channels=self.channels)
            self.apm = AudioProcessingModule(config=apm_stream_config)

            # Configure AEC
            if self.apm_aecm_enable:
                self.apm.enable_aecm(True)
                self.get_logger().info("AECM (Mobile AEC) enabled.")
            else:
                self.apm.enable_aec(True) # Standard AEC
                # The python binding might abstract specific AEC version's config
                # For AEC3, this would be the way:
                self.apm.set_aec_config({'suppression_level': self.apm_aec_suppression_level})
                self.get_logger().info(f"Standard AEC enabled with suppression level: {self.apm_aec_suppression_level}")

            # Configure Noise Suppression
            self.apm.enable_ns(True)
            self.apm.set_ns_config(NSConfig(level=self.apm_ns_level)) # Use NSConfig class

            # Configure Voice Activity Detection
            self.apm.enable_vad(True)
            self.apm.set_vad_config(VADConfig(likelihood=self.apm_vad_likelihood)) # Use VADConfig class
            self.get_logger().info("WebRTC APM initialized and configured (AEC, NS, VAD).")

        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize WebRTC APM: {e}. Node cannot function.", exc_info=True)
            # Consider rclpy.shutdown() or raising an error that main() can catch for graceful exit
            raise RuntimeError("WebRTC APM initialization failed") from e

        # VAD state logic based on 10ms APM frames
        self.num_padding_frames = int(round(self.vad_padding_ms / self.apm_frame_duration_ms))
        self.num_silence_frames_trigger = int(round(self.vad_silence_ms / self.apm_frame_duration_ms))
        self.get_logger().info(f"VAD derived (APM): Padding Frames={self.num_padding_frames}, Silence Trigger Frames={self.num_silence_frames_trigger}")

        # --- State Variables ---
        self.pa_record = None
        self.pa_play = None
        self.input_thread = None
        self.output_thread = None
        self.stop_event = threading.Event()
        self.pulse = None
        self._pulse_cm = None

        self.vad_triggered = False
        self.silence_frames_count = 0 # Renamed from silence_chunks_count
        max_speech_duration_ms = 10000 # Max duration before forced publish (optional)
        max_speech_frames = int(round(max_speech_duration_ms / self.apm_frame_duration_ms))
        # Buffer holds 10ms int16 numpy arrays of *cleaned* audio (including padding)
        self.speech_frame_buffer = collections.deque(maxlen=self.num_padding_frames + max_speech_frames)

        self.playback_queue = queue.Queue()
        self.is_playing = False

        # --- Initialize PulseAudio Client ---
        try:
             self._pulse_cm = contextlib.ExitStack()
             self.pulse = self._pulse_cm.enter_context(pulsectl.Pulse('audio-interface-node-webrtc'))
             self.get_logger().info("PulseAudio client initialized via pulsectl.")
        except Exception as e: # Catching general exception as pulsectl can raise various ones
            self.get_logger().fatal(f"Failed to connect to PulseAudio server: {e}. Node cannot function.", exc_info=True)
            if self._pulse_cm: self._pulse_cm.close()
            raise RuntimeError("PulseAudio connection failed") from e

        # pasimple audio formats: APM expects 16-bit internally
        self.pa_input_format = pa.PA_SAMPLE_S16LE # Changed from FLOAT32LE
        self.pa_output_format = pa.PA_SAMPLE_S16LE # Already S16LE, good

        # --- Device Setup ---
        self._list_devices()
        self.target_input_device_name = self._get_pa_device_name(self.input_device_param, is_input=True)
        self.target_output_device_name = self._get_pa_device_name(self.output_device_param, is_input=False)
        self.get_logger().info(f"Resolved Input Device: '{self.target_input_device_name or 'Default'}', Resolved Output Device: '{self.target_output_device_name or 'Default'}'")
        self._set_output_volume(self.target_output_device_name, self.output_volume)

        # --- Publishers/Subscribers ---
        self.vad_audio_pub = self.create_publisher(VadAudio, 'vad_audio', 10)
        self.playback_state_pub = self.create_publisher(Bool, 'playback_state', 10)
        # self.stop_playback_pub = self.create_publisher(Bool, '/stop_playback', 10) # Not used if only subscribing

        self.playback_audio_sub = self.create_subscription(
            VadAudio, 'playback_audio', self._playback_audio_callback, 10)
        self.stop_playback_sub = self.create_subscription(
            Bool, '/stop_playback', self._stop_playback_callback, 10)

        # --- Start Audio Threads ---
        if self.pulse and self.apm: # Ensure APM also initialized
            self._start_listening()
            self._start_playback_stream()
            self._publish_playback_state()
            self.get_logger().info('Audio Interface Node with WebRTC APM initialized.')
        else:
            self.get_logger().error("PulseAudio client or WebRTC APM not available. Audio threads not started.")


    # --- Device Listing and Selection (pulsectl) -UNCHANGED ---
    def _list_devices(self):
        """Logs available PulseAudio sources and sinks using pulsectl."""
        if not self.pulse:
            self.get_logger().error("Cannot list devices, PulseAudio client not available.")
            return
        self.get_logger().info("Available PulseAudio Devices:")
        try:
            self.get_logger().info("--- Input Sources ---")
            sources = self.pulse.source_list()
            if not sources:
                self.get_logger().info("  No input sources found.")
            else:
                for src in sources:
                    self.get_logger().info(f"  Index: {src.index}, Name: '{src.name}', Desc: '{src.description}'")

            self.get_logger().info("--- Output Sinks ---")
            sinks = self.pulse.sink_list()
            if not sinks:
                self.get_logger().info("  No output sinks found.")
            else:
                for sink in sinks:
                     self.get_logger().info(f"  Index: {sink.index}, Name: '{sink.name}', Desc: '{sink.description}'")

            server_info = self.pulse.server_info()
            self.get_logger().info(f"--- Defaults ---")
            self.get_logger().info(f"  Default Source: '{server_info.default_source_name}'")
            self.get_logger().info(f"  Default Sink: '{server_info.default_sink_name}'")

        except pulsectl.PulseError as e:
            self.get_logger().error(f"Could not query PulseAudio devices: {e}")
        except Exception as e: # General exception for robustness
            self.get_logger().error(f"Unexpected error listing PulseAudio devices: {e}", exc_info=True)

    def _get_pa_device_name(self, requested_name, is_input):
        """Finds the PulseAudio device name based on requested name/substring or returns default."""
        if not self.pulse:
            self.get_logger().error("Cannot get device name, PulseAudio client not available.")
            return None

        device_type = "source" if is_input else "sink"
        default_name = None
        try:
            server_info = self.pulse.server_info()
            default_name = server_info.default_source_name if is_input else server_info.default_sink_name

            if not requested_name:
                self.get_logger().info(f"Using default {device_type}: '{default_name}'")
                return default_name

            device_list = self.pulse.source_list() if is_input else self.pulse.sink_list()
            found_device = None

            for device in device_list:
                if requested_name == device.name:
                    found_device = device
                    self.get_logger().info(f"Found exact match for {device_type}: '{device.name}'")
                    break
            
            if not found_device:
                req_lower = requested_name.lower()
                for device in device_list:
                    desc_lower = device.description.lower() if device.description else ""
                    if req_lower in device.name.lower() or (desc_lower and req_lower in desc_lower):
                        if found_device is None:
                             found_device = device
                             self.get_logger().info(f"Found partial match for {device_type} '{requested_name}' -> '{device.name}' (Desc: '{device.description}')")
                        else:
                             self.get_logger().warning(f"Multiple partial matches for '{requested_name}', using first: '{found_device.name}'. Another: '{device.name}'")
                             break 
            
            if found_device:
                return found_device.name
            else:
                self.get_logger().warning(f"Could not find requested {device_type} '{requested_name}'. Falling back to default: '{default_name}'")
                return default_name

        except pulsectl.PulseError as e:
            self.get_logger().error(f"PulseAudio error querying {device_type} '{requested_name or 'default'}': {e}. Using default '{default_name}'.")
            return default_name
        except Exception as e: # General exception
            self.get_logger().error(f"Unexpected error finding {device_type} '{requested_name or 'default'}': {e}. Using default '{default_name}'.", exc_info=True)
            return default_name

    def _set_output_volume(self, sink_name_or_index, volume_fraction):
        """Sets the output volume for a specific sink using pulsectl."""
        if not self.pulse:
            self.get_logger().error("Cannot set volume, PulseAudio client not available.")
            return

        try:
            sink_info = self.pulse.get_sink_by_name(sink_name_or_index) if isinstance(sink_name_or_index, str) else self.pulse.sink_info(sink_name_or_index)
            if sink_info:
                self.pulse.volume_set_all_chans(sink_info, volume_fraction)
                self.get_logger().info(f"Output volume set to {volume_fraction:.2f} for sink '{sink_info.name}' (Index: {sink_info.index})")
            else:
                self.get_logger().error(f"Could not find sink '{sink_name_or_index}' to set volume.")
        except pulsectl.PulseError as e:
            self.get_logger().error(f"PulseAudio error setting volume for '{sink_name_or_index}': {e}")
        except Exception as e: # General exception
            self.get_logger().error(f"Unexpected error setting volume for '{sink_name_or_index}': {e}", exc_info=True)

    # --- Core Functionality: Input Thread and VAD/AEC ---
    def _input_thread_loop(self):
        self.get_logger().info("Input thread started (WebRTC APM).")
        # chunk_size_bytes is self.apm_bytes_per_frame, calculated in __init__

        stream_name = 'record-webrtc'
        server_name = None

        try:
            with contextlib.closing(pa.PaSimple(pa.PA_STREAM_RECORD,
                                               self.pa_input_format, # Should be PA_SAMPLE_S16LE
                                               self.channels,        # Should be 1
                                               self.sample_rate,
                                               __name__,
                                               stream_name,
                                               server_name,
                                               self.target_input_device_name
                                               )) as self.pa_record:
                self.get_logger().info(f"pasimple record stream started on source: {self.target_input_device_name or 'default'} for APM")

                while not self.stop_event.is_set():
                    try:
                        # Read one 10ms frame
                        mic_frame_bytes = self.pa_record.read(self.apm_bytes_per_frame)

                        if not mic_frame_bytes or len(mic_frame_bytes) != self.apm_bytes_per_frame:
                            self.get_logger().warning(f"Input stream read {len(mic_frame_bytes)} bytes, expected {self.apm_bytes_per_frame}. End of stream or error.")
                            break

                        # --- WebRTC APM Processing ---
                        # process_reverse_stream() is called in the output thread
                        # process_stream() performs AEC, NS, and prepares for VAD
                        cleaned_frame_bytes = self.apm.process_stream(mic_frame_bytes)
                        
                        # Get VAD decision from APM
                        is_voice = self.apm.is_voice_present()

                        # Convert cleaned bytes to int16 numpy array for buffer
                        # (APM output is also 16-bit S16LE bytes)
                        cleaned_frame_np = np.frombuffer(cleaned_frame_bytes, dtype=np.int16)

                        # --- VAD State Machine (using APM's 10ms frames) ---
                        if not self.vad_triggered:
                            self.speech_frame_buffer.append(cleaned_frame_np) # Buffer for padding
                            if is_voice:
                                self.vad_triggered = True
                                self.silence_frames_count = 0
                                self.get_logger().info("APM VAD: Speech started.")
                                # Barge-in: If playing, signal to stop playback
                                if self.is_playing:
                                    self.get_logger().info("Barge-in detected! Requesting playback stop.")
                                    # Send a stop message. The playback_audio_sub will also see this.
                                    stop_msg = Bool()
                                    stop_msg.data = True
                                    # Publish to internal topic or call _stop_playback_callback directly
                                    self._stop_playback_callback(stop_msg) # More direct
                        else: # vad_triggered is True
                            self.speech_frame_buffer.append(cleaned_frame_np)
                            if is_voice:
                                self.silence_frames_count = 0
                            else: # Silence frame during speech
                                self.silence_frames_count += 1
                                # self.get_logger().debug(f"APM VAD: Silence frame ({self.silence_frames_count}/{self.num_silence_frames_trigger})")
                                if self.silence_frames_count >= self.num_silence_frames_trigger:
                                    self.get_logger().debug("APM VAD: Speech ended (silence threshold reached).")
                                    self._publish_speech_segment()
                                    self.vad_triggered = False
                                    self.silence_frames_count = 0
                                    # speech_frame_buffer retains padding due to deque maxlen

                    except pa.PaSimpleError as e:
                        if not self.stop_event.is_set():
                            self.get_logger().error(f"pasimple read error: {e}")
                        break
                    except Exception as e:
                        if not self.stop_event.is_set():
                            self.get_logger().error(f"Error in input loop: {e}", exc_info=True)
                        continue
        except pa.PaSimpleError as e:
             if not self.stop_event.is_set():
                 self.get_logger().error(f"Failed to start pasimple record stream: {e}")
        except Exception as e:
            if not self.stop_event.is_set():
                self.get_logger().error(f"Unexpected error setting up input stream: {e}", exc_info=True)
        finally:
            self.pa_record = None
            self.get_logger().info("Input thread finished.")

    def _publish_speech_segment(self):
        if not self.speech_frame_buffer:
            self.get_logger().warning("Attempted to publish empty speech segment.")
            return

        # speech_frame_buffer contains deque of int16 numpy arrays (10ms each)
        speech_data_int16 = np.concatenate(list(self.speech_frame_buffer))

        if self.debug_save_vad_wav:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                wav_filename = os.path.join(self.debug_wav_output_dir, f"vad_apm_{timestamp}.wav")
                with wave.open(wav_filename, 'wb') as wf:
                    wf.setnchannels(self.channels) # Should be 1
                    wf.setsampwidth(self.apm_bytes_per_sample) # Should be 2
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(speech_data_int16.tobytes())
                self.get_logger().debug(f"Saved APM VAD audio segment to {wav_filename}")
            except Exception as e:
                self.get_logger().error(f"Failed to write VAD WAV file '{wav_filename}': {e}", exc_info=True)

        audio_list_int16 = speech_data_int16.flatten().tolist()
        vad_msg = VadAudio()
        vad_msg.header.stamp = self.get_clock().now().to_msg()
        vad_msg.header.frame_id = "audio_input_apm"
        vad_msg.audio_data = audio_list_int16
        vad_msg.sample_rate = float(self.sample_rate)

        self.vad_audio_pub.publish(vad_msg)
        self.get_logger().info(f"Published APM speech segment with {len(vad_msg.audio_data)} samples.")
        # Deque handles maintaining padding due to maxlen

    def _publish_playback_state(self):
        msg = Bool()
        msg.data = self.is_playing
        self.playback_state_pub.publish(msg)

    def _playback_audio_callback(self, msg: VadAudio):
        # This callback puts audio data (bytes) onto the playback_queue
        # The output thread will pick it up, feed to APM's process_reverse_stream, then play.
        if msg.sample_rate != self.sample_rate: # Using unified self.sample_rate
            self.get_logger().warning(
                f"Received audio for playback with mismatched sample rate: "
                f"{msg.sample_rate} Hz, expected {self.sample_rate} Hz. Skipping."
            )
            return
        try:
            # Audio data in VadAudio msg is list of int16. Convert to bytes.
            audio_np = np.array(msg.audio_data, dtype=np.int16)
            audio_bytes = audio_np.tobytes()

            # Chunk into 10ms frames for APM process_reverse_stream and playback
            num_frames_in_msg = len(audio_bytes) // self.apm_bytes_per_frame
            for i in range(num_frames_in_msg):
                frame_to_queue = audio_bytes[i*self.apm_bytes_per_frame : (i+1)*self.apm_bytes_per_frame]
                if len(frame_to_queue) == self.apm_bytes_per_frame: # Ensure full frame
                    self.playback_queue.put(frame_to_queue)
            
            # Handle potential partial last frame if audio_bytes is not multiple of apm_bytes_per_frame
            remaining_bytes = len(audio_bytes) % self.apm_bytes_per_frame
            if remaining_bytes > 0:
                partial_frame = audio_bytes[num_frames_in_msg*self.apm_bytes_per_frame:]
                padded_frame = partial_frame.ljust(self.apm_bytes_per_frame, b'\0') # Pad with silence
                self.playback_queue.put(padded_frame)
                self.get_logger().debug(f"Padded last playback frame from {remaining_bytes} to {self.apm_bytes_per_frame} bytes.")

            self.get_logger().debug(f"Added audio (chunked into {num_frames_in_msg + (1 if remaining_bytes else 0)} APM frames) to playback queue.")
        except Exception as e:
            self.get_logger().error(f"Error processing playback audio message: {e}", exc_info=True)

    def _stop_playback_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Received external stop playback command.")
            if self.is_playing:
                with self.playback_queue.mutex:
                    self.playback_queue.queue.clear()
                # Drain any potential items put after clear but before lock? Unlikely.
                while not self.playback_queue.empty(): # Additional clear just in case
                    try: self.playback_queue.get_nowait()
                    except queue.Empty: break
                self.get_logger().info("Playback queue cleared by stop command.")
                # The output thread will see the empty queue and stop.
                # If output thread is blocked on pa_play.write(), it needs to complete that write.
                # A more forceful stop might involve closing/reopening pa_play stream if pasimple allows,
                # or relying on stop_event for the thread to exit its loop.
                # For now, clearing queue is the primary mechanism.
            else:
                 self.get_logger().info("Stop command received but not currently playing.")

    def _start_listening(self):
        if self.input_thread is not None and self.input_thread.is_alive():
            self.get_logger().warning("Input thread/stream is already running.")
            return
        if not self.pulse:
             self.get_logger().error("Cannot start listening, PulseAudio client not available.")
             return
        if self.target_input_device_name is None and self.input_device_param: # Check resolved name
             self.get_logger().error(f"Cannot start listening, failed to resolve input device '{self.input_device_param}'.")
             return

        self.get_logger().info(f"Attempting to start input thread for source: {self.target_input_device_name or 'default'}")
        self.input_thread = threading.Thread(target=self._input_thread_loop, daemon=True)
        self.input_thread.start()

    def _output_thread_loop(self):
        self.get_logger().info("Output thread started (WebRTC APM).")
        stream_name = 'playback-webrtc'
        server_name = None

        # Latency/buffer attributes for pasimple playback
        # Target latency for pasimple stream (not directly for APM, but for PA buffering)
        target_latency_ms = 50 # Smaller latency might be better for responsiveness
        tlength_bytes = int(self.sample_rate * self.channels * self.apm_bytes_per_sample * target_latency_ms / 1000)
        # These params are for pasimple's internal buffering, not directly APM's processing delay (which is minimal)
        maxlength_bytes = -1
        prebuf_bytes = -1
        minreq_bytes = -1 # Should be at least apm_bytes_per_frame or multiple

        try:
            with contextlib.closing(pa.PaSimple(pa.PA_STREAM_PLAYBACK,
                                               self.pa_output_format, # S16LE
                                               self.channels,         # 1
                                               self.sample_rate,
                                               __name__,
                                               stream_name,
                                               server_name,
                                               self.target_output_device_name,
                                               tlength=tlength_bytes,
                                               maxlength=maxlength_bytes,
                                               minreq=minreq_bytes, # Ensure this is small enough, e.g. apm_bytes_per_frame
                                               prebuf=prebuf_bytes
                                               )) as self.pa_play:
                self.get_logger().info(f"pasimple playback stream started on sink: {self.target_output_device_name or 'default'}")

                while not self.stop_event.is_set():
                    try:
                        # Get one 10ms audio frame (bytes) from queue
                        far_end_frame_bytes = self.playback_queue.get(timeout=0.1) # Timeout to check stop_event

                        if far_end_frame_bytes: # Should be self.apm_bytes_per_frame
                            if not self.is_playing:
                                self.get_logger().info(f"Playback started.")
                                # VAD threshold change logic removed, rely on AEC
                                self.is_playing = True
                                self._publish_playback_state()

                            # --- Feed to APM as far-end/render stream for AEC ---
                            self.apm.process_reverse_stream(far_end_frame_bytes)
                            
                            # Write data to PulseAudio
                            self.pa_play.write(far_end_frame_bytes)
                            self.playback_queue.task_done()

                    except queue.Empty:
                        if self.is_playing: # Was playing, now queue is empty
                             self.get_logger().info("Playback queue empty, draining PA buffer...")
                             try:
                                self.pa_play.drain()
                                self.get_logger().info("Playback buffer drained.")
                             except pa.PaSimpleError as drain_err:
                                 if "Broken pipe" not in str(drain_err) and not self.stop_event.is_set():
                                     self.get_logger().warning(f"Error draining playback stream: {drain_err}")
                             except Exception as drain_exc:
                                 if not self.stop_event.is_set():
                                     self.get_logger().error(f"Unexpected error draining playback stream: {drain_exc}", exc_info=True)
                             finally:
                                if self.is_playing: # Ensure state is updated
                                    self.get_logger().info(f"Playback stopped.")
                                    self.is_playing = False
                                    # VAD threshold restoration removed
                                    self._publish_playback_state()
                        continue # Continue loop to wait for more data or stop_event

                    except pa.PaSimpleError as e:
                        if not self.stop_event.is_set() and "Broken pipe" not in str(e):
                             self.get_logger().error(f"pasimple write error: {e}")
                        if self.is_playing:
                           self.is_playing = False; self._publish_playback_state()
                        break # Exit loop on stream error
                    except Exception as e:
                        if not self.stop_event.is_set():
                            self.get_logger().error(f"Error in output loop: {e}", exc_info=True)
                        if self.is_playing:
                           self.is_playing = False; self._publish_playback_state()
                        break
        except pa.PaSimpleError as e:
            if not self.stop_event.is_set():
                self.get_logger().error(f"Failed to start pasimple playback stream: {e}")
        except Exception as e:
            if not self.stop_event.is_set():
                self.get_logger().error(f"Unexpected error setting up output stream: {e}", exc_info=True)
        finally:
            if self.is_playing: # Ensure state reset on thread exit
                self.is_playing = False
                self._publish_playback_state()
            self.pa_play = None
            self.get_logger().info("Output thread finished.")

    def _start_playback_stream(self):
        if self.output_thread is not None and self.output_thread.is_alive():
            self.get_logger().warning("Output stream/thread already started.")
            return
        if not self.pulse:
             self.get_logger().error("Cannot start playback, PulseAudio client not available.")
             return
        if self.target_output_device_name is None and self.output_device_param: # Check resolved name
             self.get_logger().error(f"Cannot start playback, failed to resolve output device '{self.output_device_param}'.")
             return

        self.get_logger().info(f"Attempting to start output thread for sink: {self.target_output_device_name or 'default'}")
        self.output_thread = threading.Thread(target=self._output_thread_loop, daemon=True)
        self.output_thread.start()

    def destroy_node(self):
        self.get_logger().info("Shutting down Audio Interface Node (WebRTC APM)...")
        self.stop_event.set()

        if self.input_thread is not None and self.input_thread.is_alive():
            self.get_logger().info("Stopping audio input thread...")
            self.input_thread.join(timeout=2.0)
            if self.input_thread.is_alive(): self.get_logger().warning("Input thread did not stop cleanly.")
            else: self.get_logger().info("Input thread stopped.")
        self.input_thread = None

        if self.output_thread is not None and self.output_thread.is_alive():
            self.get_logger().info("Stopping audio output thread...")
            self.output_thread.join(timeout=2.0)
            if self.output_thread.is_alive(): self.get_logger().warning("Output thread did not stop cleanly.")
            else: self.get_logger().info("Output thread stopped.")
        self.output_thread = None

        if self._pulse_cm:
            self.get_logger().info("Closing PulseAudio client connection...")
            try:
                self._pulse_cm.close() # This calls __exit__ on the pulsectl.Pulse object
                self.get_logger().info("PulseAudio client closed.")
            except Exception as e:
                 self.get_logger().error(f"Error closing PulseAudio client: {e}", exc_info=True)
        self.pulse = None # Clear reference
        self.apm = None # Clear APM reference

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AudioInterfaceNode()
        if node.input_thread is None or not node.input_thread.is_alive(): # Check if input thread started
             # Check if APM failed initialization explicitly (could be another reason)
             if not hasattr(node, 'apm') or node.apm is None:
                 node.get_logger().error("WebRTC APM failed to initialize or input thread failed. Shutting down.")
             else:
                 node.get_logger().error("Audio input thread failed to initialize. Shutting down.")
             # Node will be destroyed in finally
             return # Exit main
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info("KeyboardInterrupt received.")
    except RuntimeError as e:
        # Catches init errors like PA connection or APM init fail
        print(f"FATAL ERROR during node initialization: {e}", file=sys.stderr)
        if node and hasattr(node, 'get_logger'): # Logger might not be available if error is very early
            node.get_logger().fatal(f"Node initialization failed: {e}", exc_info=True)
    except Exception as e: # Catch any other unexpected error during spin
        if node: node.get_logger().fatal(f"Unhandled exception in main: {e}", exc_info=True)
        else: print(f"FATAL Unhandled exception in main (node not fully initialized): {e}", file=sys.stderr)
    finally:
        if node and rclpy.ok():
             node.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()

if __name__ == '__main__':
    main()