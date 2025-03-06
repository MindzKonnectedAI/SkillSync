// app/agent-demo/page.tsx
"use client";

import { useEffect, useState } from "react";
// import { createLangGraphAgent } from "@/lib/agent";
import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";


export default function AgentDemo() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const thread = useStream<{ messages: Message[] }>({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    messagesKey: "messages",
  });
  
  console.log("thread", thread);
  console.log("data", data);

  useEffect(() => {
    // console.log("Agent Demo", createLangGraphAgent().then(res => console.log(res)));
    fetch("/api/agent")
      .then((res) => {
        if (!res.ok) {
          throw new Error("Network response was not ok");
        }
        return res.json();
      })
      .then((data) => {
        setData(data.result);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching agent output:", err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div>
      <h1>LangGraph Agent Output</h1>
      <div>
        {thread.messages.map((message) => (
          <div key={message.id}>{message.content as string}</div>
        ))}
      </div>
      {loading && <p>Loading...</p>}
      {error && <p>Error: {error}</p>}
      {data && <pre>{JSON.stringify(data, null, 2)}</pre>}
    </div>
  );
}
