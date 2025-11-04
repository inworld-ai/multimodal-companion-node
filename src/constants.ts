export const DEFAULT_VOICE_ID = 'Dennis';
export const DEFAULT_LLM_MODEL_NAME = 'gpt-4o-mini';
export const DEFAULT_PROVIDER = 'openai';
export const DEFAULT_TTS_MODEL_ID = 'inworld-tts-1';
export const DEFAULT_VAD_MODEL_PATH = '../assets/models/silero_vad.onnx';
export const INPUT_SAMPLE_RATE = 16000;
export const TTS_SAMPLE_RATE = 24000;
export const PAUSE_DURATION_THRESHOLD_MS = 650;
export const MIN_SPEECH_DURATION_MS = 200;
export const FRAME_PER_BUFFER = 1024;
export const SPEECH_THRESHOLD = 0.5;
// v0.51
// export const TEXT_CONFIG = {
//   maxNewTokens: 500,
//   maxPromptLength: 1000,
//   repetitionPenalty: 1,
//   topP: 0.5,
//   temperature: 0.1,
//   frequencyPenalty: 0,
//   presencePenalty: 0,
//   stopSequences: ['\n'],
// };
export const TEXT_CONFIG = {
  max_new_tokens: 2500,
  max_prompt_length: 100,
  repetition_penalty: 1,
  top_p: 1,
  temperature: 1,
  frequency_penalty: 0,
  presence_penalty: 0,
  stop_sequences: [] as string[],
};
export const WS_APP_PORT = 4000;
