import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async rewrites() {
    const backend = process.env.STORE_ASSISTANT_API_URL?.replace(/\/$/, "");
    if (backend) {
      return [{ source: "/v1/:path*", destination: `${backend}/v1/:path*` }];
    }
    if (process.env.VERCEL) {
      return [{ source: "/v1/:path*", destination: "/api/v1/:path*" }];
    }
    return [];
  },
};

export default nextConfig;
