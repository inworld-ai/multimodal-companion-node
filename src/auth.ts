import crypto from 'crypto';
import type { IncomingMessage } from 'http';
import type { Request, Response, NextFunction } from 'express';
import { parse } from 'url';

// In-memory nonce store to prevent replay (simple TTL management)
const nonceTimestamps = new Map<string, number>();
const NONCE_TTL_MS = 5 * 60 * 1000; // 5 minutes

export interface ParsedAuthHeader {
  apiKey: string;
  dateTime: string; // YYYYMMDDHHmmss
  nonce: string;
  signature: string; // hex lowercase
}

function cleanupExpiredNonces(now: number) {
  for (const [nonce, ts] of nonceTimestamps.entries()) {
    if (now - ts > NONCE_TTL_MS) {
      nonceTimestamps.delete(nonce);
    }
  }
}

function parseAuthorizationHeader(header: string | undefined): ParsedAuthHeader | null {
  if (!header) return null;
  // Expected: "IW1-HMAC-SHA256 ApiKey=...,DateTime=...,Nonce=...,Signature=..."
  const scheme = 'IW1-HMAC-SHA256';
  const trimmed = header.trim();
  if (!trimmed.startsWith(scheme)) return null;
  const parts = trimmed.substring(scheme.length).trim();
  const kvs = parts.split(',').map((p) => p.trim());
  const map: Record<string, string> = {};
  for (const kv of kvs) {
    const [k, v] = kv.split('=');
    if (k && v) map[k] = v;
  }
  const apiKey = map['ApiKey'];
  const dateTime = map['DateTime'];
  const nonce = map['Nonce'];
  const signature = map['Signature'];
  if (!apiKey || !dateTime || !nonce || !signature) return null;
  return { apiKey, dateTime, nonce, signature };
}

// Match the C# logic: start with key = "IW1" + secret, then for each string do HMAC-SHA256 over the string
// with the current key, and set key = digest. Final signature = hex(lowercase) of the final key bytes.
function generateSignatureCSharpCompatible(values: string[], secret: string): string {
  // Use Uint8Array to avoid Buffer generic type mismatch across TS libs
  let key: Uint8Array = Buffer.from('IW1' + secret, 'utf8');
  for (const s of values) {
    const h = crypto.createHmac('sha256', key);
    h.update(Buffer.from(s, 'utf8'));
    const digest: Uint8Array = h.digest();
    key = digest; // overwrite key like the C# code does
  }
  return Buffer.from(key).toString('hex');
}

function isDateTimeSkewAcceptable(dateTime: string): boolean {
  // dateTime format: YYYYMMDDHHmmss (UTC)
  const m = dateTime.match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$/);
  if (!m) return false;
  const [_, y, mo, d, h, mi, s] = m;
  const dt = Date.UTC(Number(y), Number(mo) - 1, Number(d), Number(h), Number(mi), Number(s));
  const now = Date.now();
  const skew = Math.abs(now - dt);
  return skew <= NONCE_TTL_MS; // within 5 minutes
}

function getClientCredsFromInworldEnv(): { apiKey: string; apiSecret: string } | null {
  const b64 = (process.env.INWORLD_API_KEY || '').trim();
  if (!b64) return null;
  try {
    const decoded = Buffer.from(b64, 'base64').toString('utf8');
    const idx = decoded.indexOf(':');
    if (idx <= 0) return null;
    const apiKey = decoded.slice(0, idx);
    const apiSecret = decoded.slice(idx + 1);
    if (!apiKey || !apiSecret) return null;
    return { apiKey, apiSecret };
  } catch {
    return null;
  }
}

export function verifyIncomingRequest(req: IncomingMessage | Request): { ok: boolean; error?: string } {
  // Accept auth from header or from query param "auth" encoded as the same header value
  // For WS upgrade, we get IncomingMessage; for HTTP we get Express Request
  const headers = (req as IncomingMessage).headers || (req as Request).headers;
  const url = (req as IncomingMessage).url || (req as Request).url;

  let header = headers['authorization'] as string | undefined;
  if (!header && url) {
    const { query } = parse(url, true);
    const qAuth = typeof (query as any).auth === 'string' ? (query as any).auth : undefined;
    if (qAuth) {
      // For query param, spaces are often encoded as '+' by some clients; restore and decode.
      try {
        header = decodeURIComponent(qAuth.replace(/\+/g, ' '));
      } catch {
        header = qAuth;
      }
    }
  }

  const parsed = parseAuthorizationHeader(header);
  if (!parsed) return { ok: false, error: 'Missing or invalid Authorization' };

  const creds = getClientCredsFromInworldEnv();
  if (!creds) return { ok: false, error: 'Server not configured for client auth (INWORLD_API_KEY missing/invalid)' };
  if (parsed.apiKey !== creds.apiKey) return { ok: false, error: 'ApiKey mismatch' };
  if (!isDateTimeSkewAcceptable(parsed.dateTime)) return { ok: false, error: 'DateTime skew too large' };

  // check nonce replay
  const now = Date.now();
  cleanupExpiredNonces(now);
  if (nonceTimestamps.has(parsed.nonce)) return { ok: false, error: 'Nonce replayed' };

  // Replicate C# signature
  const METHOD = 'ai.inworld.engine.v1.SessionTokens/GenerateSessionToken';
  const TAIL = 'iw1_request';
  // For studioServer value, Unity side uses their actual server value; here we accept host from request headers
  const host = (headers['host'] as string) || '';
  const values = [parsed.dateTime, host, METHOD, parsed.nonce, TAIL];
  const expected = generateSignatureCSharpCompatible(values, creds.apiSecret);
  if (expected !== parsed.signature) return { ok: false, error: 'Signature mismatch' };

  // Accept once
  nonceTimestamps.set(parsed.nonce, now);
  return { ok: true };
}

export function authMiddleware(req: Request, res: Response, next: NextFunction) {
  const result = verifyIncomingRequest(req);
  if (!result.ok) return res.status(401).json({ error: result.error || 'Unauthorized' });
  next();
}


