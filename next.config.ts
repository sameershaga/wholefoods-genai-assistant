import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async rewrites() {
    const backend = process.env.STORE_ASSISTANT_API_URL?.replace(/\/$/, "");
    return backend
      ? [{ source: "/v1/:path*", destination: `${backend}/v1/:path*` }]
      : [];
  },
};

export default nextConfig;
