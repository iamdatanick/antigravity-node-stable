import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          "vendor-react": ["react", "react-dom", "react-router"],
          "vendor-query": ["@tanstack/react-query", "zustand"],
          "vendor-charts": ["recharts"],
          "vendor-editor": ["@codemirror/view", "@codemirror/state", "@codemirror/lang-sql", "@codemirror/theme-one-dark"],
          "vendor-terminal": ["@xterm/xterm", "@xterm/addon-fit"],
          "vendor-graph": ["cytoscape"],
          "vendor-highlight": ["highlight.js", "marked"],
        },
      },
    },
  },
  server: {
    port: 1055,
    proxy: {
      "/api/ws": {
        target: "http://localhost:8080",
        ws: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/api/budget": {
        target: "http://localhost:4000",
        rewrite: (path) => path.replace(/^\/api\/budget/, ""),
      },
      "/api": {
        target: "http://localhost:8080",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
