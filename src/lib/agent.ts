// agent.ts

// IMPORTANT - Add your API keys here. Be careful not to publish them.
// process.env.NEXT_PUBLIC_OPENAI_API_KEY =
//     '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';
// process.env.TAVILY_API_KEY = 'tvly-...';

// import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from '@langchain/openai';
import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';
import { createReactAgent } from '@langchain/langgraph/prebuilt';

export const createLangGraphAgent = async () => {
    // Define the tools for the agent to use
    // const agentTools = [new TavilySearchResults({ maxResults: 3 })];
    const agentModel = new ChatOpenAI({ temperature: 0, model: "gpt-4o-mini", apiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY });

    // Initialize memory to persist state between graph runs
    const agentCheckpointer = new MemorySaver();
    const agent = createReactAgent({
        llm: agentModel,
        tools: [],
        checkpointSaver: agentCheckpointer,
    });

    // Now it's time to use!
    const agentFinalState = await agent.invoke(
        { messages: [new HumanMessage('what is the current weather in sf')] },
        { configurable: { thread_id: '42' } }
    );

    console.log(
        agentFinalState.messages[agentFinalState.messages.length - 1].content
    );

    const agentNextState = await agent.invoke(
        { messages: [new HumanMessage('what about ny')] },
        { configurable: { thread_id: '42' } }
    );

    // console.log(
    //     agentNextState.messages[agentNextState.messages.length - 1].content
    // );

    return agentNextState;
};
