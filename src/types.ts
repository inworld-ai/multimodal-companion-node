import { AudioChunkInterface } from '@inworld/runtime/common';

export enum EVENT_TYPE {
  TEXT = 'text',
  AUDIO = 'audio',
  AUDIO_SESSION_END = 'audioSessionEnd',
  IMAGE_CHAT = 'imageChat',
}

export enum AUDIO_SESSION_STATE {
  PROCESSING = 'PROCESSING',
  ACTIVE = 'ACTIVE',
  INACTIVE = 'INACTIVE',
}

export interface ChatMessage {
  id: string;
  role: string;
  content: string;
}

export interface Agent {
  id: string;
  name: string;
  description: string;
  motivation: string;
  knowledge?: string[];
}

export interface TextInput {
  key: string;
  text: string;
  interactionId: string;
}

export interface ImageChatInput {
  key: string;
  text: string;
  image: string; // base64 encoded image
  voiceId?: string; // Optional voice ID for TTS, defaults to system default
  interactionId: string;
}

export interface AudioInput {
  key: string;
  audio: AudioChunkInterface;
  state: State;
  interactionId: string;
}

export interface State {
  agent: Agent;
  userName: string;
  messages: ChatMessage[];
  imageUrl: string;
}

export interface Connection {
  state: State;
  ws: any;
}

export interface PromptInput {
  agent: Agent;
  messages: ChatMessage[];
  userName: string;
  userQuery: string;
}

export interface CreateGraphPropsInterface {
  apiKey: string;
  dialogPromptTemplate: string;
  graphVisualizationEnabled: boolean;
  connections: {
    [key: string]: Connection;
  };
}
