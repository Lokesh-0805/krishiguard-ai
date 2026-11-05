import type { NextConfig } from "next";

// All /api/* calls (and the FastAPI docs) are proxied to the backend so the browser only talks to one origin.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  output: "standalone",
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` },
      { source: "/docs", destination: `${BACKEND_URL}/docs` },
      { source: "/openapi.json", destination: `${BACKEND_URL}/openapi.json` },
    ];
  },
};

export default nextConfig;
