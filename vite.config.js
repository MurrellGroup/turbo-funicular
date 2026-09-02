import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  plugins: [basicSsl()],
  server: {
    host: "0.0.0.0",
    port: 8791,
    strictPort: true,
    hmr: false,
  },
  preview: {
    host: "0.0.0.0",
    port: 8791,
    strictPort: true,
  },
  build: {
    target: "es2022",
  },
});
