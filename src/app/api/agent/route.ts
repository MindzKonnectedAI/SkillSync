// app/api/agent/route.ts
import { NextResponse } from "next/server";

import { createLangGraphAgent } from "@/lib/agent";
import { supervisorGithubAgent } from "@/lib/githubAgent";
import { HumanMessage } from '@langchain/core/messages';
import { LangChainAdapter } from 'ai';
import { graph } from "@/lib/githubAgent";

export async function POST(request: Request) {
  try {
    const messages = await request.json();
    // console.log("message", message)
    // console.log("request", request)
    // const res = request
    // Invoke the LangGraph agent and get the result.
    // const result = await createLangGraphAgent();
    const result = await supervisorGithubAgent(messages);

    console.log("result111111111111111111111111111111111111111111", result)
    return NextResponse.json({ result });

    // const finalState = graph.streamEvents({
    //   messages: [new HumanMessage(message)],
    // }, { streamMode: "messages", version: "v2" });

    // const finalState = graph.streamEvents({
    //   messages: [new HumanMessage({content: messages})],
    // }, { streamMode: "messages", version: "v2" });
  

    // return LangChainAdapter.toDataStreamResponse(finalState);
    // const finalState = graph.streamEvents(
    //   { input },
    //   { version: "v2" },
    // );


  } catch (error) {
    console.error("Agent error:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
