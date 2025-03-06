'use server';

import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { AgentExecutor, createToolCallingAgent } from 'langchain/agents';
import { pull } from 'langchain/hub';
import { createStreamableValue } from 'ai/rsc';
import { WikipediaQueryRun } from '@langchain/community/tools/wikipedia_query_run';
import { graph } from '@/lib/githubAgent';
import { HumanMessage } from '@langchain/core/messages';
import { compiledSuperGraph } from './superAgent';


// export async function runAgent(input: string) {
//     'use server';
//     const tool = new WikipediaQueryRun({
//         topKResults: 1,
//         maxDocContentLength: 100,
//     });

//     const stream = createStreamableValue();
//     (async () => {
//         const tools = [tool];
//         const prompt = await pull<ChatPromptTemplate>(
//             'hwchase17/openai-tools-agent'
//         );

//         const llm = new ChatOpenAI({ model: 'gpt-4o-mini', temperature: 0 });

//         const agent = createToolCallingAgent({
//             llm,
//             tools,
//             prompt,
//         });

//         const agentExecutor = new AgentExecutor({ agent, tools });

//         const streamingEvents = graph.streamEvents(
//             {
//                 messages: [
//                     new HumanMessage({
//                         content: input,
//                     }),
//                 ],
//             },
//             { version: 'v2' }
//         );

//         for await (const item of streamingEvents) {
//             stream.update(JSON.parse(JSON.stringify(item, null, 2)));
//         }

//         stream.done();
//     })();

//     return { streamData: stream.value };
// }

import { AbortController } from 'abort-controller'; // if needed

export async function runAgent(input: string) {
    'use server';
    const stream = createStreamableValue();
    const abortController = new AbortController();

    (async () => {
        try {
            // Pass the signal if the API supports it.
            // const streamingEvents = graph.streamEvents(
            //     {
            //         messages: [
            //             new HumanMessage({
            //                 content: input,
            //             }),
            //         ]
            //     },
            //     { version: 'v2' }
            // );

            // for await (const item of streamingEvents) {
            //     stream.update(JSON.parse(JSON.stringify(item, null, 2)));
            // }

            const streamResults = compiledSuperGraph.stream(
                {
                    messages: [
                        new HumanMessage( input ),
                    ],
                },
                { recursionLimit: 50 }
            );
        
            for await (const output of await streamResults) {

                console.log("output", output);
                stream.update(JSON.parse(JSON.stringify(output, null, 2)));
                // if (!output?.__end__) {
                //     console.log(output);
                //     console.log('----');
                // }
            }        

            stream.done();
        } catch (error) {
            console.error('Error:', error);
        } finally {
            // Abort the controller to ensure all associated listeners are removed
            abortController.abort();
        }
    })();

    return { streamData: stream.value };
}
