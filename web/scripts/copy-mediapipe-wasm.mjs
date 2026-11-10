// Copies the MediaPipe WASM runtime into public/ so it is served locally
// instead of from a CDN. Runs on `npm install`.
import { cpSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const from = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const to = join(root, "public", "mediapipe", "wasm");

mkdirSync(to, { recursive: true });
cpSync(from, to, { recursive: true });
console.log("MediaPipe WASM copied to public/mediapipe/wasm");
