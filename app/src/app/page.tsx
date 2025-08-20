"use client";

import Output from "@/components/Output";
import TextArea from "@/components/TextArea";
import { type ChatOutput } from "@/types";
import { useState } from "react";

export default function Home() {
  const [outputs, setOutputs] = useState<ChatOutput[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  return (
    <div className="min-h-screen bg-[#131313]">
      <div
        className={`container mx-auto px-6 pt-16 pb-32 min-h-screen ${
          outputs.length === 0 && "flex items-center justify-center"
        }`}
      >
        <div className="w-full">
          {outputs.length === 0 && (
            <div className="text-center space-y-8 max-w-2xl mx-auto">
              {/* Main Heading */}
              <div className="space-y-3">
                <h1 className="text-5xl font-light text-white tracking-tight">
                  What do you want to know?
                </h1>
                <p className="text-lg text-gray-400">
                  AI-powered search with real-time answers
                </p>
              </div>

              {/* Simple Project Description */}
              <div className="text-center text-gray-500 text-sm space-y-2">
                <p>A Perplexity clone built with LangChain, Groq, and SerpAPI</p>
                <p>Features web search, calculations, and streaming responses</p>
              </div>
            </div>
          )}

          <TextArea
            setIsGenerating={setIsGenerating}
            isGenerating={isGenerating}
            outputs={outputs}
            setOutputs={setOutputs}
          />

          {outputs.map((output, i) => {
            return <Output key={i} output={output} />;
          })}
        </div>
      </div>
    </div>
  );
}