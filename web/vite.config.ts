import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// No backend HTTP server to proxy — comments arrive over Tauri events. The
// dev server only needs to serve the web bundle for the Tauri webview.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
});