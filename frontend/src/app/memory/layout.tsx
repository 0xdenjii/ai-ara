import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Memory - ARA",
};

export default function MemoryLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
