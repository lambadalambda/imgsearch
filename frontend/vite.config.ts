import { defineConfig, loadEnv } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  // Load IMGSEARCH_API_KEY from the repo-root .env so dev API calls authenticate.
  const env = loadEnv(mode, path.resolve(dirname, ".."), "");
  const apiKey = (env.IMGSEARCH_API_KEY || process.env.IMGSEARCH_API_KEY || "").replace(
    /^['"]|['"]$/g,
    "",
  );

  const proxyTarget = env.IMGSEARCH_PROXY_TARGET || "http://localhost:8081";

  const proxyConfig = {
    target: proxyTarget,
    changeOrigin: true,
    configure: (proxy: any) => {
      proxy.on("proxyReq", (proxyReq: any) => {
        if (apiKey) {
          proxyReq.setHeader("X-Imgsearch-API-Key", apiKey);
        }
      });
    },
  };

  return {
    plugins: [svelte(), tailwindcss()],
    resolve: {
      alias: {
        "@": path.resolve(dirname, "src"),
      },
    },
    build: {
      // Build directly into the Go embed root so `go build` picks up the latest assets.
      outDir: path.resolve(dirname, "../internal/webui/atelier/dist"),
      // Keep `.gitkeep` so the Go //go:embed directive still has at least one file.
      emptyOutDir: false,
      sourcemap: false,
      target: "es2022",
    },
    server: {
      port: 5173,
      strictPort: false,
      proxy: {
        "/api": proxyConfig,
        "/media": proxyConfig,
        "/healthz": proxyConfig,
        "/legacy": proxyConfig,
      },
    },
  };
});
