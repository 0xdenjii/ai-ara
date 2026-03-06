import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Upload - ARA",
};

export default function UploadLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
