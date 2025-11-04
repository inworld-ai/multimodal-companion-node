import {
  CustomNode,
  ProcessContext,
  RemoteSTTNode,
  ProxyNode,
  GraphBuilder,
  Graph
} from '@inworld/runtime/graph';
import { GraphTypes } from '@inworld/runtime/common';
import * as os from 'os';
import * as path from 'path';

import {
  AudioInput,
  CreateGraphPropsInterface
} from './types';

export class STTGraph {
  executor: InstanceType<typeof Graph>;

  private constructor({
    executor,
  }: {
    executor: InstanceType<typeof Graph>;
  }) {
    this.executor = executor;
  }

  destroy() {
    this.executor.stopExecutor();
    this.executor.cleanupAllExecutions();
    this.executor.destroy();
  }

  static async create(props: CreateGraphPropsInterface) {
    const {
      apiKey,
    } = props;

    const postfix = '-with-audio-input';

    const graphName = `character-chat${postfix}`;
    const graph = new GraphBuilder(graphName);

    class AudioFilterNode extends CustomNode {
      process(_context: ProcessContext, input: AudioInput): GraphTypes.Audio {
        return new GraphTypes.Audio({
          data: input.audio.data,
          sampleRate: input.audio.sampleRate,
        });
      }
    }

    // start node to pass the audio input to the audio filter node
    const audioInputNode = new ProxyNode();
    
    const audioFilterNode = new AudioFilterNode();
    const sttNode = new RemoteSTTNode();



    // Wish app would actually report an error when a node is missing at graph compilation stage, not execution
    graph
    .addNode(audioInputNode)
    .addNode(audioFilterNode)
    .addNode(sttNode)
    .addEdge(audioInputNode, audioFilterNode)
    .addEdge(audioFilterNode, sttNode)
    .setStartNode(audioInputNode);

    graph.setEndNode(sttNode);

    const executor = graph.build();
    if (props.graphVisualizationEnabled) {
      console.log(
        'The Graph visualization has started..If you see any fatal error after this message, pls disable graph visualization.',
      );
      const graphPath = path.join(os.tmpdir(), `${graphName}.png`);

      await executor.visualize(graphPath);
    }

    return new STTGraph({
      executor,
    });
  }
}
