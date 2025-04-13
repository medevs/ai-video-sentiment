import Link from "next/link";
import Navbar from "~/components/navbar";

export default function HomePage() {
  return (
    <>
      <Navbar />
      <main className="flex min-h-screen flex-col bg-white">
        {/* Hero Section */}
        <section className="relative overflow-hidden bg-gradient-to-br from-blue-50 to-indigo-100 py-20">
          <div className="absolute -top-24 -right-24 h-64 w-64 rounded-full bg-blue-200 opacity-50 blur-3xl"></div>
          <div className="absolute bottom-0 left-0 h-64 w-64 rounded-full bg-purple-200 opacity-50 blur-3xl"></div>
          
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid items-center gap-12 md:grid-cols-2">
              <div className="flex flex-col space-y-8">
                <div>
                  <h1 className="mb-4 text-4xl font-extrabold tracking-tight text-gray-900 sm:text-5xl md:text-6xl">
                    Unlock the <span className="text-blue-600">emotions</span> in your videos
                  </h1>
                  <p className="text-lg text-gray-600 md:text-xl">
                    VibeScan uses advanced AI to analyze sentiment and emotions in video, audio, and text content, giving you powerful insights into audience reactions.
                  </p>
                </div>
                
                <div className="flex flex-col space-y-4 sm:flex-row sm:space-x-4 sm:space-y-0">
                  <Link 
                    href="/signup" 
                    className="inline-flex items-center justify-center rounded-md bg-blue-600 px-5 py-3 text-base font-medium text-white shadow-md transition hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  >
                    Get Started Free
                  </Link>
                  <Link 
                    href="/dashboard/demo" 
                    className="inline-flex items-center justify-center rounded-md border border-gray-300 bg-white px-5 py-3 text-base font-medium text-gray-700 shadow-sm transition hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  >
                    Try Demo
                  </Link>
                </div>
              </div>
              
              <div className="relative mx-auto h-[400px] w-full max-w-lg rounded-lg shadow-xl">
                <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 opacity-90"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center text-white">
                    <div className="mb-4 flex justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="h-16 w-16">
                        <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                      </svg>
                    </div>
                    <p className="text-lg font-medium">Video Analysis Demo</p>
                    <p className="mt-2 text-sm opacity-80">Click to see VibeScan in action</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="mb-16 text-center">
              <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
                Multimodal Sentiment Analysis
              </h2>
              <p className="mt-4 text-lg text-gray-600">
                Our advanced AI analyzes multiple dimensions of your content
              </p>
            </div>

            <div className="grid gap-8 md:grid-cols-3">
              {/* Video Analysis */}
              <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm transition hover:shadow-md">
                <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-md bg-blue-100 text-blue-600">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="h-6 w-6">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h1.5C5.496 19.5 6 18.996 6 18.375m-3.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-1.5A1.125 1.125 0 0118 18.375M20.625 4.5H3.375m17.25 0c.621 0 1.125.504 1.125 1.125M20.625 4.5h-1.5C18.504 4.5 18 5.004 18 5.625m-3.75 0v1.5c0 .621-.504 1.125-1.125 1.125M3.375 4.5c-.621 0-1.125.504-1.125 1.125M3.375 4.5h1.5C5.496 4.5 6 5.004 6 5.625m-3.75 0v1.5c0 .621.504 1.125 1.125 1.125m0 0h1.5m-1.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125M19.125 12h1.5m0 0c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h1.5m14.25 0h1.5" />
                  </svg>
                </div>
                <h3 className="mb-2 text-xl font-semibold text-gray-900">Video Analysis</h3>
                <p className="text-gray-600">
                  Detect facial expressions, gestures, and visual cues to understand emotional responses in video content.
                </p>
              </div>

              {/* Audio Analysis */}
              <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm transition hover:shadow-md">
                <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-md bg-purple-100 text-purple-600">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="h-6 w-6">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z" />
                  </svg>
                </div>
                <h3 className="mb-2 text-xl font-semibold text-gray-900">Audio Analysis</h3>
                <p className="text-gray-600">
                  Analyze tone, pitch, and vocal patterns to identify emotional states and sentiment in spoken content.
                </p>
              </div>

              {/* Text Analysis */}
              <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm transition hover:shadow-md">
                <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-md bg-indigo-100 text-indigo-600">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="h-6 w-6">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
                  </svg>
                </div>
                <h3 className="mb-2 text-xl font-semibold text-gray-900">Text Analysis</h3>
                <p className="text-gray-600">
                  Process transcripts and captions to extract sentiment, key phrases, and emotional context from written content.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section className="bg-gray-50 py-20">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="mb-16 text-center">
              <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
                How VibeScan Works
              </h2>
              <p className="mt-4 text-lg text-gray-600">
                Simple process, powerful insights
              </p>
            </div>

            <div className="grid gap-8 md:grid-cols-4">
              {/* Step 1 */}
              <div className="text-center">
                <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-blue-600 text-white">
                  <span className="text-xl font-bold">1</span>
                </div>
                <h3 className="mb-2 text-xl font-semibold text-gray-900">Upload</h3>
                <p className="text-gray-600">
                  Upload your video content to our secure platform
                </p>
              </div>

              {/* Step 2 */}
              <div className="text-center">
                <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-blue-600 text-white">
                  <span className="text-xl font-bold">2</span>
                </div>
                <h3 className="mb-2 text-xl font-semibold text-gray-900">Analyze</h3>
                <p className="text-gray-600">
                  Our AI processes video, audio, and text simultaneously
                </p>
              </div>

              {/* Step 3 */}
              <div className="text-center">
                <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-blue-600 text-white">
                  <span className="text-xl font-bold">3</span>
                </div>
                <h3 className="mb-2 text-xl font-semibold text-gray-900">Visualize</h3>
                <p className="text-gray-600">
                  View detailed sentiment analysis with intuitive visualizations
                </p>
              </div>

              {/* Step 4 */}
              <div className="text-center">
                <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-blue-600 text-white">
                  <span className="text-xl font-bold">4</span>
                </div>
                <h3 className="mb-2 text-xl font-semibold text-gray-900">Act</h3>
                <p className="text-gray-600">
                  Use insights to improve content and engagement
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="bg-gradient-to-r from-blue-600 to-indigo-700 py-16 text-white">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-3xl text-center">
              <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
                Ready to unlock the emotions in your videos?
              </h2>
              <p className="mt-4 text-lg">
                Join thousands of content creators and marketers using VibeScan to understand their audience better.
              </p>
              <div className="mt-8 flex flex-col justify-center space-y-4 sm:flex-row sm:space-x-4 sm:space-y-0">
                <Link 
                  href="/signup" 
                  className="inline-flex items-center justify-center rounded-md bg-white px-5 py-3 text-base font-medium text-blue-700 shadow-md transition hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-blue-700"
                >
                  Get Started Free
                </Link>
                <Link 
                  href="/contact" 
                  className="inline-flex items-center justify-center rounded-md border border-white bg-transparent px-5 py-3 text-base font-medium text-white shadow-md transition hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-blue-700"
                >
                  Contact Sales
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-gray-200 bg-white py-12">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="md:flex md:items-center md:justify-between">
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-md bg-gradient-to-br from-blue-600 to-purple-600 text-white">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    className="h-5 w-5"
                  >
                    <path d="M5.507 4.048A3 3 0 017.785 3h8.43a3 3 0 012.278 1.048l1.722 2.008A4.533 4.533 0 0019.5 6h-15c-.243 0-.482.02-.715.056l1.722-2.008z" />
                    <path
                      fillRule="evenodd"
                      d="M1.5 10.5a3 3 0 013-3h15a3 3 0 110 6h-15a3 3 0 01-3-3zm15 0a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm2.25.75a.75.75 0 100-1.5.75.75 0 000 1.5zM4.5 15a3 3 0 100 6h15a3 3 0 100-6h-15zm11.25 3.75a.75.75 0 100-1.5.75.75 0 000 1.5zM19.5 18a.75.75 0 11-1.5 0 .75.75 0 011.5 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <span className="text-lg font-semibold text-gray-900">VibeScan</span>
              </div>
              <p className="mt-4 text-sm text-gray-500 md:mt-0">
                &copy; {new Date().getFullYear()} VibeScan. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </main>
    </>
  );
}
