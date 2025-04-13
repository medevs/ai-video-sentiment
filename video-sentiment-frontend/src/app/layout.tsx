import "~/styles/globals.css";

import { type Metadata } from "next";
import { Geist } from "next/font/google";
import SessionProvider from "~/components/client/SessionProvider";

export const metadata: Metadata = {
  title: "VibeScan",
  description: "VibeScan - Video Sentiment Analysis.",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geist.variable}`}>
      <body>
        <SessionProvider>
          {children}
        </SessionProvider>
      </body>
    </html>
  );
}
