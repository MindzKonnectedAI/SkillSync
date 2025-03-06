// app/agent-demo/page.tsx
"use client";

import { useEffect, useState } from "react";

export default function AgentDemo() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
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
      {loading && <p>Loading...</p>}
      {error && <p>Error: {error}</p>}
      {data && <pre>{JSON.stringify(data, null, 2)}</pre>}
    </div>
  );
}
