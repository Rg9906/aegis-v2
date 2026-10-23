import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The live console talks to the Python engine (python -m aegis.console, port 8765) under /api
const engine = {
  "/api": { target: "http://127.0.0.1:8765", changeOrigin: true, ws: true },
};

// Relative base so the built site works from any folder or static host
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { proxy: engine },
  preview: { proxy: engine },
});
