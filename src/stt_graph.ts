import {
  CustomNode,
  ProcessContext,
  RemoteSTTNode,
  ProxyNode,
  GraphBuilder,
  Graph,
} from '@inworld/runtime/graph';
import { GraphTypes } from '@inworld/runtime/common';
import * as os from 'os';
import * as path from 'path';

import { AudioInput, CreateGraphPropsInterface } from './types';

export class STTGraph {
  executor: Graph;

  private constructor({ executor }: { executor: Graph }) {
    this.executor = executor;
  }

  async destroy() {
    if (this.executor) {
      try {
        await this.executor.stop();
      } catch (error: unknown) {
        console.error('Error stopping executor (non-fatal):', error);
      }
    }
  }

  static async create(props: CreateGraphPropsInterface) {
    const { apiKey: _apiKey } = props;

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
      const graphPath = path.join(os.tmpdir(), `${graphName}.png`);
      console.log(
        `The Graph visualization will be saved to ${graphPath}. If you see any fatal error after this message, pls disable graph visualization.`
      );
      // TODO: visualize() method should be added back in rc17
      // Once available, uncomment: await executor.visualize(graphPath);
      // Note: Currently visualization can be done using the Inworld CLI: npx @inworld/cli graph visualize <file>
    }

    return new STTGraph({
      executor,
    });
  }
}
