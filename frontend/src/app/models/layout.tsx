import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Models - ARA",
};

export default function ModelsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
