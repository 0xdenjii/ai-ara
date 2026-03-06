import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Knowledge Graph - ARA",
};

export default function GraphLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
