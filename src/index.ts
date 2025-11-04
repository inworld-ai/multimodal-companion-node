import express from 'express';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import fsPromises from 'fs/promises';
import path from 'path';
import { GraphBuilder, RemoteLLMChatNode } from '@inworld/runtime/graph';
import { TEXT_CONFIG, DEFAULT_VAD_MODEL_PATH } from './constants';
import dotenv from 'dotenv';
import { ContentInterface } from '@inworld/runtime';
import { VADFactory } from '@inworld/runtime/primitives/vad';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { parse } from 'url';
import { RawData } from 'ws';
import { MessageHandler } from './message_handler';
import { STTGraph } from './stt_graph';
import { authMiddleware, verifyIncomingRequest } from './auth';
dotenv.config();

const upload = multer({ dest: 'uploads/' });
const app = express();
const server = createServer(app);
const webSocket = new WebSocketServer({ noServer: true });
const PORT = process.env.PORT || 3000;

// Add CORS headers for all requests
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

let vadClient: any;
let sttGraph: STTGraph;
const connections: { [key: string]: { ws?: any } } = {};
// Short-lived WS tokens issued after HTTP auth; validated during WS upgrade
const wsTokens: { [sessionKey: string]: { token: string; expiresAt: number } } = {};
const WS_TOKEN_TTL_MS = 5 * 60 * 1000; // 5 minutes

function createMessages(
  prompt: string,
  imageUrl?: string,
) {
  const systemMessage = {
    role: 'system',
    content:
      'You are a helpful assistant.',
  };

  let userMessage;
  if (imageUrl) {
    userMessage = {
      role: 'user',
      content: [
        {
          type: 'text' as const,
          text: prompt,
        },
        {
          type: 'image' as const,
          image_url: {
            url: imageUrl,
            detail: 'high',
          },
        },
      ],
    };
  } else {
    userMessage = {
      role: 'user',
      content: prompt,
    };
  }

  return {
    messages: [systemMessage, userMessage],
  };
}

// WebSocket connection handler
webSocket.on('connection', (ws, request) => {
  const { query } = parse(request.url!, true);
  const key = query.key?.toString();

  if (!key) {
    ws.close(4000, 'Session key required');
    return;
  }

  if (!connections[key]) {
    connections[key] = {};
  }

  connections[key].ws = ws;
  console.log(`WebSocket connected for session: ${key}`);

  ws.on('error', console.error);

  const messageHandler = new MessageHandler(
    sttGraph,
    vadClient,
    (data: any) => ws.send(JSON.stringify(data))
  );

  ws.on('message', (data: RawData) => {
    messageHandler.handleMessage(data, key);
  });

  ws.on('close', () => {
    console.log(`WebSocket disconnected for session: ${key}`);
    if (connections[key]) {
      delete connections[key];
    }
  });
});

// Serve test HTML files
function resolveStaticFile(relativePath: string) {
  // 1) Try alongside the compiled file (dist)
  const inSameDir = path.join(__dirname, relativePath);
  if (fs.existsSync(inSameDir)) return inSameDir;
  // 2) Try project root (when running from dist, parent is root)
  const inParent = path.join(__dirname, '..', relativePath);
  if (fs.existsSync(inParent)) return inParent;
  // 3) Fallback to CWD
  const inCwd = path.join(process.cwd(), relativePath);
  return inCwd;
}

app.get('/test-image', (req, res) => {
  res.sendFile(resolveStaticFile('examples/test-image-chat.html'));
});

app.get('/test-audio', (req, res) => {
  res.sendFile(resolveStaticFile('examples/test-audio.html'));
});

// Create WebSocket session endpoint (protected)
app.post('/create-session', authMiddleware, (req, res) => {
  const sessionKey = uuidv4();
  connections[sessionKey] = {};
  const wsToken = uuidv4();
  wsTokens[sessionKey] = { token: wsToken, expiresAt: Date.now() + WS_TOKEN_TTL_MS };
  res.json({ sessionKey, wsToken });
});

// Development helper: issue short-lived access tokens for test pages without exposing auth in browser
// Enable via ALLOW_TEST_CLIENT=true. DO NOT enable in production.
app.get('/get_access_token', (req, res) => {
  if (process.env.ALLOW_TEST_CLIENT !== 'true') {
    return res.status(403).json({ error: 'Disabled. Set ALLOW_TEST_CLIENT=true for local testing.' });
  }
  const sessionKey = uuidv4();
  connections[sessionKey] = {};
  const wsToken = uuidv4();
  wsTokens[sessionKey] = { token: wsToken, expiresAt: Date.now() + WS_TOKEN_TTL_MS };
  res.json({ sessionKey, wsToken });
});

// Protect chat endpoint as well
app.post('/chat', authMiddleware, upload.single('image'), async (req, res) => {
  let imageFile = req.file;
  try {
    console.log("Received request:", req.body);
    const prompt = req.body.prompt;

    if (!prompt) {
      return res.status(400).json({ error: 'Missing prompt' });
    }

    let imageUrl;
    if (imageFile) {
      const imageBuffer = fs.readFileSync(imageFile.path);
      const base64Image = imageBuffer.toString('base64');
      const mimeType = imageFile.mimetype;
      imageUrl = `data:${mimeType};base64,${base64Image}`;
    }

    const llmNode = new RemoteLLMChatNode({
      id: uuidv4() + '_llm_node',      
        provider: 'google', // 'openai'
        modelName: 'gemini-2.5-flash-lite', // 'gpt-4o-mini'
        stream: false,
        textGenerationConfig: TEXT_CONFIG,      
    });

    const executor = new GraphBuilder({ 
      id: 'http_llm_chat_graph',
      apiKey: process.env.INWORLD_API_KEY!
    })
      .addNode(llmNode)
      .setStartNode(llmNode)
      .setEndNode(llmNode)
      .build();

    const input = {
      ...createMessages(prompt, imageUrl),
    };

    console.log("Executing graph with input:", input);
    const executionResult = await executor.start(input, { executionId: uuidv4() });

    let output = '';
    console.log("Processing output stream...");
    try {
      for await (const result of executionResult.outputStream) {
        if (result.data && (result.data as ContentInterface).content) {
          output = (result.data as ContentInterface).content;
        }
        console.log("Received result:", result);
      }
    } finally {
      try {
        executor.closeExecution(executionResult.outputStream);
      } catch (e) {
        console.error('Error closing execution:', e);
      }
    }

    res.json({ response: output });
  } catch (err: any) {
    console.error(err);
    res.status(500).json({ error: err?.message || 'Internal server error' });
  } finally {
    // Clean up uploaded file asynchronously
    if (imageFile?.path) {
      fsPromises.unlink(imageFile.path).catch((e) => {
        // Ignore ENOENT (file doesn't exist) errors, log others
        if (e.code !== 'ENOENT') {
          console.error('Error deleting file:', e);
        }
      });
    }
  }
});

// Handle WebSocket upgrade
server.on('upgrade', async (request, socket, head) => {
  const { pathname } = parse(request.url!);

  if (pathname === '/ws') {
    // DEBUG: log every WS upgrade hit to help diagnose connectivity/auth issues
    try {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      console.log('WS upgrade hit:', request.url, request.headers?.host);
    } catch {}
    // Prefer short-lived wsToken issued by /create-session; fallback to full Authorization if missing
    let allowed = false;
    let denyReason = '';
    try {
      const { query } = parse(request.url!, true);
      const key = typeof (query as any).key === 'string' ? (query as any).key : undefined;
      const token = typeof (query as any).wsToken === 'string' ? (query as any).wsToken : undefined;
      if (key && token) {
        const record = wsTokens[key];
        if (record && record.token === token && record.expiresAt >= Date.now()) {
          allowed = true;
          // one-time use
          delete wsTokens[key];
          console.log('WS wsToken auth success for:', request.url);
        } else {
          denyReason = 'Invalid or expired wsToken';
        }
      }
    } catch (e) {
      denyReason = 'wsToken parse error';
    }

    if (!allowed) {
      const v = verifyIncomingRequest(request);
      if (!v.ok) {
        try {
          console.warn('WS auth failed:', denyReason || v.error, request.url);
        } catch {}
        socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
        socket.destroy();
        return;
      }
      try { console.log('WS Authorization auth success for:', request.url); } catch {}
    }
    webSocket.handleUpgrade(request, socket, head, (ws) => {
      webSocket.emit('connection', ws, request);
    });
  } else {
    socket.destroy();
  }
});

server.listen(PORT, async () => {
  try {
    // Initialize VAD client
    const vadModelPath = process.env.VAD_MODEL_PATH || path.join(__dirname, DEFAULT_VAD_MODEL_PATH);
    vadClient = await VADFactory.createLocal({
      modelPath: vadModelPath,
    });
    console.log('VAD client initialized');

    // Initialize STT Graph (requires TTS params even for STT-only usage)
    sttGraph = await STTGraph.create({
      apiKey: process.env.INWORLD_API_KEY!,
      dialogPromptTemplate: '',
      graphVisualizationEnabled: false,
      connections: {},
    });
    console.log('STT Graph initialized');
    
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`WebSocket available at ws://localhost:${PORT}/ws?key=<session_key>`);
  } catch (error) {
    console.error('Failed to initialize server:', error);
    process.exit(1);
  }
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('Shutting down server...');
  if (sttGraph) {
    sttGraph.destroy();
  }
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  console.log('Shutting down server...');
  if (sttGraph) {
    sttGraph.destroy();
  }
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

