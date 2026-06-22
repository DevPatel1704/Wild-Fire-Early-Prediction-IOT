import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API = process.env.VITE_API || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": { target: API, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") },
      "/eval": { target: API, changeOrigin: true },
      "/ws":  { target: API.replace("http", "ws"), ws: true },
    },
  },
});
