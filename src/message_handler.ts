import { v4 } from 'uuid';
import { RawData } from 'ws';
import type { VAD } from '@inworld/runtime/primitives/vad';
import {
  GraphBuilder,
  RemoteLLMChatNode,
  RemoteTTSNode,
  TextChunkingNode,
  Graph,
  GraphOutputStream,
} from '@inworld/runtime/graph';

import { GraphTypes, TTSOutputStreamIterator } from '@inworld/runtime/common';
// eslint-disable-next-line @typescript-eslint/no-require-imports
const WavEncoder = require('wav-encoder');

import {
  FRAME_PER_BUFFER,
  INPUT_SAMPLE_RATE,
  MIN_SPEECH_DURATION_MS,
  PAUSE_DURATION_THRESHOLD_MS,
  SPEECH_THRESHOLD,
  TEXT_CONFIG,
  DEFAULT_TTS_MODEL_ID,
  DEFAULT_VOICE_ID,
  TTS_SAMPLE_RATE,
} from './constants';
import { AudioInput, EVENT_TYPE, TextInput, ImageChatInput } from './types';
import { EventFactory } from './event_factory';
import { STTGraph } from './stt_graph';

// Observability: track active executes to verify no unintended concurrency
let STT_ACTIVE_EXECUTIONS = 0;
let IMAGECHAT_ACTIVE_EXECUTIONS = 0;
function logActive(tag: 'STT' | 'IMAGE', value: number, interactionId: string) {
  // Minimal, high-signal log for tracking concurrent executes
  console.log(
    `[ActiveExecutions] ${tag}=${value} (interactionId=${interactionId})`
  );
}

export class MessageHandler {
  private INPUT_SAMPLE_RATE = INPUT_SAMPLE_RATE;
  private FRAME_PER_BUFFER = FRAME_PER_BUFFER;
  private PAUSE_DURATION_THRESHOLD_MS = PAUSE_DURATION_THRESHOLD_MS;
  private MIN_SPEECH_DURATION_MS = MIN_SPEECH_DURATION_MS;

  private pauseDuration = 0;
  private isCapturingSpeech = false;
  private speechBuffer: number[] = [];
  private processingQueue: (() => Promise<void>)[] = [];
  private isProcessing = false;

  // Shared ImageChat executor (reuse until voiceId changes)
  private imageChatExecutor: InstanceType<typeof Graph> | null = null;
  private imageChatCurrentVoiceId: string | null = null;

  private async ensureImageChatExecutor(voiceId: string) {
    if (this.imageChatExecutor && this.imageChatCurrentVoiceId === voiceId)
      return;
    // Rebuild when first time or voiceId changed
    try {
      this.imageChatExecutor?.cleanupAllExecutions?.();
    } catch {
      // Ignore cleanup errors
    }
    try {
      this.imageChatExecutor?.stopExecutor?.();
    } catch {
      // Ignore stop errors
    }
    try {
      this.imageChatExecutor?.destroy?.();
    } catch {
      // Ignore destroy errors
    }
    this.imageChatExecutor = null;

    // Create TTS node with the target voice
    const ttsNode = new RemoteTTSNode({
      id: `tts-node`,
      speakerId: voiceId,
      modelId: DEFAULT_TTS_MODEL_ID,
      sampleRate: TTS_SAMPLE_RATE,
      temperature: 0.8,
      speakingRate: 1.0,
    });

    // Create LLM node with streaming enabled
    const llmNode = new RemoteLLMChatNode({
      id: `llm-node`,
      provider: 'google',
      modelName: 'gemini-2.5-flash-lite',
      stream: true, // Enable streaming for TTS
      textGenerationConfig: TEXT_CONFIG,
    });

    const textChunkingNode = new TextChunkingNode({
      id: `text-chunking-node`,
    });

    // Build LLM -> TTS pipeline once
    this.imageChatExecutor = new GraphBuilder({
      id: `image-chat-tts`,
      apiKey: process.env.INWORLD_API_KEY!,
    })
      .addNode(llmNode)
      .addNode(ttsNode)
      .addNode(textChunkingNode)
      .addEdge(llmNode, textChunkingNode)
      .addEdge(textChunkingNode, ttsNode)
      .setStartNode(llmNode)
      .setEndNode(ttsNode)
      .build();

    this.imageChatCurrentVoiceId = voiceId;
  }

  constructor(
    private graph: STTGraph,
    private vadClient: VAD | null,
    private send: (data: Record<string, unknown>) => void
  ) {}

  async handleMessage(data: RawData, key: string) {
    const message = JSON.parse(data.toString());
    const interactionId = v4();

    switch (message.type) {
      case EVENT_TYPE.TEXT: {
        const input = {
          text: message.text,
          interactionId,
          key,
        } as TextInput;

        this.addToQueue(() =>
          this.executeGraph({
            key,
            input,
            interactionId,
            graph: this.graph,
          })
        );

        break;
      }

      case EVENT_TYPE.IMAGE_CHAT: {
        const imageChatInput = {
          text: message.text,
          image: message.image,
          voiceId: message.voiceId,
          interactionId,
          key,
        } as ImageChatInput;

        this.addToQueue(() =>
          this.executeImageChat({
            key,
            input: imageChatInput,
            interactionId,
          })
        );

        break;
      }

      case EVENT_TYPE.AUDIO: {
        const audioBuffer: number[] = [];
        for (let i = 0; i < message.audio.length; i++) {
          Object.values(message.audio[i]).forEach((value) => {
            if (typeof value === 'number') {
              audioBuffer.push(value);
            }
          });
        }

        if (audioBuffer.length >= this.FRAME_PER_BUFFER && this.vadClient) {
          const audioChunk = {
            data: audioBuffer,
            sampleRate: this.INPUT_SAMPLE_RATE,
          };
          const vadResult = await this.vadClient.detectVoiceActivity(
            audioChunk,
            SPEECH_THRESHOLD
          );

          if (this.isCapturingSpeech) {
            this.speechBuffer.push(...audioChunk.data);
            if (vadResult === -1) {
              // Already capturing speech but new chunk has no voice activity
              this.pauseDuration +=
                (audioChunk.data.length * 2000) / this.INPUT_SAMPLE_RATE;
              if (this.pauseDuration > this.PAUSE_DURATION_THRESHOLD_MS) {
                this.isCapturingSpeech = false;

                const speechDuration =
                  (this.speechBuffer.length * 2000) / this.INPUT_SAMPLE_RATE;
                if (speechDuration > this.MIN_SPEECH_DURATION_MS) {
                  console.log('speechDuration', speechDuration);
                  await this.processCapturedSpeech(key, interactionId);
                }
              }
            } else {
              // Already capturing speech and new chunk has voice activity
              this.pauseDuration = 0;
            }
          } else {
            if (vadResult !== -1) {
              // Not capturing speech but new chunk has voice activity. start capturing speech
              this.isCapturingSpeech = true;

              this.speechBuffer.push(...audioChunk.data);
              this.pauseDuration = 0;
            } else {
              // Not capturing speech and new chunk has no voice activity. do nothing
            }
          }
        }
        break;
      }

      case EVENT_TYPE.AUDIO_SESSION_END:
        this.pauseDuration = 0;
        this.isCapturingSpeech = false;

        if (this.speechBuffer.length > 0) {
          await this.processCapturedSpeech(key, interactionId);
        }

        break;
    }
  }

  private normalizeAudio(audioBuffer: number[]): number[] {
    let maxVal = 0;
    // Find maximum absolute value
    for (let i = 0; i < audioBuffer.length; i++) {
      maxVal = Math.max(maxVal, Math.abs(audioBuffer[i]));
    }

    if (maxVal === 0) {
      return audioBuffer;
    }

    // Create normalized copy
    const normalizedBuffer = [];
    for (let i = 0; i < audioBuffer.length; i++) {
      normalizedBuffer.push(audioBuffer[i] / maxVal);
    }

    return normalizedBuffer;
  }

  private async processCapturedSpeech(key: string, interactionId: string) {
    try {
      const input: AudioInput = {
        audio: {
          // Normalize to get consistent input regardless of how loud or quiet the user's microphone input is.
          // Avoid normalizing before VAD else quiet ambient sound can be amplified and trigger VAD.
          data: this.normalizeAudio(this.speechBuffer),
          sampleRate: this.INPUT_SAMPLE_RATE,
        },
        state: {
          agent: {
            id: 'demo-agent',
            name: 'Demo Agent',
            description: 'Demo STT Agent',
            motivation: 'Help with speech recognition',
          },
          userName: 'User',
          messages: [],
          imageUrl: '',
        },
        interactionId,
        key,
      } as AudioInput;

      this.speechBuffer = [];

      this.addToQueue(() =>
        this.executeGraph({
          key,
          input,
          interactionId,
          graph: this.graph,
        })
      );
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';
      console.error('Error processing captured speech:', errorMessage);
    }
  }

  private async executeImageChat({
    key: _key,
    input,
    interactionId,
  }: {
    key: string;
    input: ImageChatInput;
    interactionId: string;
  }) {
    try {
      // Use provided voiceId or default
      const voiceId = input.voiceId || DEFAULT_VOICE_ID;

      // Ensure a shared executor exists with the correct voice
      await this.ensureImageChatExecutor(voiceId);

      // Create messages similar to the HTTP chat endpoint
      const systemMessage = {
        role: 'system',
        content: 'You are a helpful assistant.',
      };

      const userMessage = {
        role: 'user',
        content: [
          {
            type: 'text' as const,
            text: input.text,
          },
          {
            type: 'image' as const,
            image_url: {
              url: input.image,
              detail: 'high',
            },
          },
        ],
      };

      const graphInput = new GraphTypes.LLMChatRequest({
        messages: [systemMessage, userMessage],
      });

      IMAGECHAT_ACTIVE_EXECUTIONS++;
      logActive('IMAGE', IMAGECHAT_ACTIVE_EXECUTIONS, interactionId);
      const executionResult = await this.imageChatExecutor!.start(graphInput, {
        executionId: v4(),
      });

      try {
        // Handle streaming TTS response
        await this.handleTTSResponse(
          executionResult.outputStream,
          interactionId
        );
      } finally {
        this.send(EventFactory.interactionEnd(interactionId));
        try {
          this.imageChatExecutor!.closeExecution(executionResult.outputStream);
        } catch {
          // Ignore close errors
        }
        IMAGECHAT_ACTIVE_EXECUTIONS--;
        logActive('IMAGE', IMAGECHAT_ACTIVE_EXECUTIONS, interactionId);
      }
    } catch (error) {
      console.error('Error in executeImageChat:', error);
      const errorPacket = EventFactory.error(
        error instanceof Error ? error : new Error(String(error)),
        interactionId
      );
      this.send(errorPacket);
    }
  }

  private async handleTTSResponse(
    outputStream: GraphOutputStream,
    interactionId: string
  ) {
    try {
      const result = await outputStream.next();
      const ttsStream = result.data as TTSOutputStreamIterator;

      if (ttsStream?.next) {
        let chunk = await ttsStream.next();

        while (!chunk.done) {
          // Send text chunk
          if (chunk.text) {
            const textPacket = EventFactory.text(chunk.text, interactionId, {
              isAgent: true,
              name: 'Agent',
            });
            this.send(textPacket);
          }

          // Send audio chunk
          if (chunk.audio && chunk.audio.data) {
            const audioBuffer = await WavEncoder.encode({
              sampleRate: chunk.audio.sampleRate,
              channelData: [new Float32Array(chunk.audio.data)],
            });

            this.send(
              EventFactory.audio(
                Buffer.from(audioBuffer).toString('base64'),
                interactionId,
                v4() // utteranceId
              )
            );
          }

          chunk = await ttsStream.next();
        }
      }
    } catch (error) {
      console.error('Error in handleTTSResponse:', error);
      const errorPacket = EventFactory.error(
        error instanceof Error ? error : new Error(String(error)),
        interactionId
      );
      this.send(errorPacket);
    }
  }

  private async executeGraph({
    key: _key,
    input,
    interactionId,
    graph,
  }: {
    key: string;
    input: TextInput | AudioInput;
    interactionId: string;
    graph: STTGraph;
  }) {
    const executor = graph.executor;
    STT_ACTIVE_EXECUTIONS++;
    logActive('STT', STT_ACTIVE_EXECUTIONS, interactionId);
    const executionResult = await executor.start(input, { executionId: v4() });

    try {
      await this.handleResponse(executionResult.outputStream, interactionId);
    } finally {
      this.send(EventFactory.interactionEnd(interactionId));
      try {
        executor.closeExecution(executionResult.outputStream);
      } catch {
        // Ignore close errors
      }
      STT_ACTIVE_EXECUTIONS--;
      logActive('STT', STT_ACTIVE_EXECUTIONS, interactionId);
    }
  }

  private async handleResponse(
    outputStream: GraphOutputStream,
    interactionId: string
  ) {
    try {
      const result = await outputStream.next();
      const sttOutput = result.data as string;

      console.log('TTS Stream:', sttOutput);
      const textPacket = EventFactory.text(sttOutput, interactionId, {
        isAgent: true,
        name: 'User',
      });

      this.send(textPacket);
    } catch (error) {
      console.error(error);
      const errorPacket = EventFactory.error(
        error instanceof Error ? error : new Error(String(error)),
        interactionId
      );
      // Ignore errors caused by empty speech.
      if (!errorPacket.error.includes('recognition produced no text')) {
        this.send(errorPacket);
      }
    }
  }

  private addToQueue(task: () => Promise<void>) {
    this.processingQueue.push(task);
    this.processQueue();
  }

  private async processQueue() {
    if (this.isProcessing) {
      return;
    }
    this.isProcessing = true;
    while (this.processingQueue.length > 0) {
      const task = this.processingQueue.shift();
      if (task) {
        try {
          await task();
        } catch (error) {
          console.error('Error processing task from queue:', error);
        }
      }
    }
    this.isProcessing = false;
  }
}
