import { tool } from '@langchain/core/tools';
import { RunnableConfig } from '@langchain/core/runnables';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { SystemMessage } from '@langchain/core/messages';
import { HumanMessage } from '@langchain/core/messages';
import { END, Annotation } from '@langchain/langgraph';
import { BaseMessage } from '@langchain/core/messages';
import { z } from 'zod';
import {
    ChatPromptTemplate,
    MessagesPlaceholder,
} from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';

const llm = new ChatOpenAI({ model: 'gpt-4o-mini', temperature: 0 });

const AgentState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
        default: () => [],
    }),
    // The agent node that last performed work
    next: Annotation<string>({
        reducer: (x, y) => y ?? x ?? END,
        default: () => END,
    }),
});

const supportedQueries = tool(
    async (query: string): Promise<string> => {
        console.log('query in supported_queries node:', query);
        return query;
    },
    {
        name: 'supported_queries',
        description:
            'Takes a natural language user query as input and only returns the parameters acceptable by the Github User Search API.',
    }
);

const queryParamGenerator = tool(
    async (query: string): Promise<string> => {
        console.log('query in query_param_generator node:', query);
        return query;
    },
    {
        name: 'query_param_generator',
        description:
            'Takes a natural language query as input and returns the appropriate query parameters.',
    }
);

// Define the input schema using Zod
const fetchUsersSchema = z.object({
    queryParam: z.string().describe('Search query for GitHub users'),
    perPage: z
        .number()
        .optional()
        .default(10)
        .describe('Number of results per page (1-100)'),
});

interface GithubUser {
    login: string;
    id: number;
    avatar_url: string;
    html_url: string;
}

interface GithubApiResponse {
    total_count: number;
    incomplete_results: boolean;
    items: GithubUser[];
}

const fetchUsers = tool(
    // Function now takes a single object parameter matching the Zod schema
    async ({
        queryParam,
        perPage = 10,
    }: z.infer<typeof fetchUsersSchema>): Promise<GithubApiResponse> => {
        try {
            perPage = Math.min(Math.max(perPage, 1), 100);
            const url = `https://api.github.com/search/users?q=${queryParam}&per_page=${perPage}`;
            const res = await fetch(url);

            if (!res.ok) {
                throw new Error(
                    `GitHub API error: ${res.status} ${res.statusText}`
                );
            }

            return await res.json();
        } catch (error) {
            console.error('Error fetching users:', error);
            throw new Error(
                'Failed to fetch GitHub users. Please check your input and try again.'
            );
        }
    },
    {
        name: 'fetch_users',
        description: 'Returns a list of GitHub users based on search query',
        schema: fetchUsersSchema, // Attach the Zod schema
    }
);

const supportedQueriesAgent = () => {
    const query_gen_system = `
    You are a specialized Query Parameters Filter Agent. Your task is to analyze a Job Description written in natural language and extract only the relevant fields supported by the GitHub User Search API. Focus exclusively on fields that are searchable via the API.

    Your output should be a list of strings containing only those fields from the job description that align with the GitHub User Search API's capabilities.

    Below are the fields supported by the GitHub User Search API:
    1. Job Title: Title or role of the candidate
    2. Location: Candidate’s location
    3. Repositories: Number of GitHub repositories the candidate has
    4. Followers: Number of GitHub followers the candidate has
    5. Number of candidates: How many candidates to retrieve
    6. Language: Programming languages used by the candidate
    7. Email: Candidate’s email address
    8. Bio: Information from the candidate's bio
    9. Username: Candidate’s GitHub username
    10. is:sponsorable: Boolean indicating whether the candidate is sponsorable
    `;

    return createReactAgent({
        llm,
        tools: [],
        // tools: [supportedQueries],
        stateModifier: new SystemMessage(query_gen_system),
    });
};

const supportedQueriesAgentNode = async (
    state: typeof AgentState.State,
    config?: RunnableConfig
) => {
    const result = await supportedQueriesAgent().invoke(state, config);
    const lastMessage = result.messages[result.messages.length - 1];
    return {
        messages: [
            new HumanMessage({
                content: lastMessage.content,
                name: 'supported_queries',
            }),
        ],
    };
};

const queryParamGeneratorAgent = () => {
    const query_gen_system = `
    Strictly follow all the rules below to generate query parameters for GitHub User Search . Ensure all rules are followed to generate accurate search queries.
    Only return the query , no extra information .
    Prefix every query you generate with 'type:user' 
    # GitHub User Search - Query Parameter Generation Rules

    The following guidelines define how to construct query parameters for the GitHub User Search. Ensure all rules are followed to generate accurate search queries.

    # Overview

    1. Search Scope: Applies to public personal GitHub accounts (not organizations).
    2. Query Components: Queries can include:
    3. Keywords: For general information like usernames, names, emails, and bios.
    4. Qualifiers: For searching specific fields.
    5. Sort Parameters: Optional but can be added for ordering results.
    6. Case Sensitivity: Keywords are case-insensitive.
    7. Result Limit: The search returns the first 1000 results, sorted by best match (by default).

    # Qualifiers & Usage

    Each qualifier targets a specific field in the GitHub user data. These cannot be mixed with regular keywords.

    user:NAME: Matches exact usernames.
    Example: user:braingain

    in:login: Searches within usernames (non-exact matches allowed).
    Example: braingain in:login

    in:email: Searches within users' email addresses.
    Example: irina in:email

    in:name: Searches within users' full names.
    Example: Irina in:name

    fullname:NAME: Similar to in:name, searches users' full names.
    Example: fullname:john smith

    location:NAME: Searches users based on location.
    Example: location:Boston

    language:NAME: Finds users based on the primary language of their public repositories.
    Example: language:python

    repos:n: Searches users by the number of public repositories.
    Example: repos:>1000

    followers:n: Searches users by the number of followers.
    Example: followers:>1000

    created:DATE: Finds users by their GitHub account creation date.
    Example: created:>2020-01-01

    is:sponsorable: Finds users with a GitHub Sponsors profile.
    Example: is:sponsorable

    sort:: Sorts users based on specific attributes.
    Example: repos:>10000 sort:followers

    # Boolean Operators
    You can combine keywords and qualifiers using Boolean operators to refine the search. Follow these rules:

    AND (implied): Combining two different qualifiers or a qualifier and a keyword automatically implies AND.
    Example: location:"San Francisco" language:python (Finds users in San Francisco who primarily use Python).

    OR: Explicitly use OR between keywords or in: qualifiers only. For other qualifiers, using the same qualifier twice implies OR.
    Example: "front-end developer" OR "ui developer"
    Example: location:"new jersey" location:"new york" (Finds users in either New Jersey or New York).

    NOT (-): Use the minus sign (-) to exclude certain terms or qualifiers.
    Example: location:iceland -location:Reykjavik (Finds users in Iceland but not Reykjavik).

    # Key Limitations & Constraints

    Character Limit: Queries must not exceed 256 characters.

    No Parentheses: Do not use parentheses in queries.

    AND/OR/NOT Limits: You cannot use more than five AND, OR, or NOT operators in a single query.
    For example: location:"silicon valley" -language:java -language:c++ -language:python -language:javascript -language:html is valid (5 negations).
    Special Notes on Combining Operators

    AND is implied for combining qualifiers and keywords but cannot be used explicitly with certain fields like location, language, etc.

    OR cannot be used explicitly between different qualifiers.
    Example: fullname:irina user:braingain is interpreted as AND, while fullname:irina OR user:braingain is invalid.

    You cannot combine keywords and qualifiers in OR statements.
    Example: language:java OR "java developer" is invalid, while language:java "java developer" is interpreted as AND.
    
    # Output format

    Always return ONLY the query parameters you generated WITHOUT any extra text . 
    Below are examples of some output that will be considered correct : 

    1. type:user AI Engineer location:India created:<2019-01-01 repos:>50
    2. type:user Blockchain developer location:Ukraine sort:followers
    3. type:user Full Stack Developer location:USA created:>2020-01-01 
    4. type:user UX Engineer location:Delhi 
    `;

    return createReactAgent({
        llm,
        tools: [],
        // tools: [queryParamGenerator],
        stateModifier: new SystemMessage(query_gen_system),
    });
};

const queryParamGeneratorAgentNode = async (
    state: typeof AgentState.State,
    config?: RunnableConfig
) => {
    const result = await queryParamGeneratorAgent().invoke(state, config);
    const lastMessage = result.messages[result.messages.length - 1];
    return {
        messages: [
            new HumanMessage({
                content: lastMessage.content,
                name: 'query_param_generator',
            }),
        ],
    };
};

const fetchUsersAgent = () => {
    const query_gen_system = `
    You are a specialized Query Parameters Filter Agent. Your task is to analyze a Job Description written in natural language and extract only the relevant fields supported by the GitHub User Search API. Focus exclusively on fields that are searchable via the API.

    Your output should be a list of strings containing only those fields from the job description that align with the GitHub User Search API's capabilities.

    Below are the fields supported by the GitHub User Search API:
    1. Job Title: Title or role of the candidate
    2. Location: Candidate’s location
    3. Repositories: Number of GitHub repositories the candidate has
    4. Followers: Number of GitHub followers the candidate has
    5. Number of candidates: How many candidates to retrieve
    6. Language: Programming languages used by the candidate
    7. Email: Candidate’s email address
    8. Bio: Information from the candidate's bio
    9. Username: Candidate’s GitHub username
    10. is:sponsorable: Boolean indicating whether the candidate is sponsorable
    `;

    return createReactAgent({
        llm,
        tools: [fetchUsers],
        stateModifier: new SystemMessage(query_gen_system),
    });
};

const fetchUsersAgentNode = async (
    state: typeof AgentState.State,
    config?: RunnableConfig
) => {
    const result = await fetchUsersAgent().invoke(state, config);
    const lastMessage = result.messages[result.messages.length - 1];
    return {
        messages: [
            new HumanMessage({
                content: lastMessage.content,
                name: 'fetch_users',
            }),
        ],
    };
};

const members = [
    'supported_queries',
    'query_param_generator',
    'fetch_users',
] as const;

const systemPrompt =
    'You are a supervisor tasked with managing a conversation between the' +
    ' following workers: {members}. Given the following user request,' +
    ' respond with the worker to act next. Each worker will perform a' +
    ' task and respond with their results and status. When finished,' +
    ' respond with FINISH.';

const options = [END, ...members];

// Define the routing function
const routingTool = {
    name: 'route',
    description: 'Select the next role.',
    schema: z.object({
        next: z.enum([END, ...members]),
    }),
};

const prompt = ChatPromptTemplate.fromMessages([
    ['system', systemPrompt],
    new MessagesPlaceholder('messages'),
    [
        'human',
        'Given the conversation above, who should act next?' +
            ' Or should we FINISH? Select one of: {options}',
    ],
]);

const formattedPrompt = await prompt.partial({
    options: options.join(', '),
    members: members.join(', '),
});

// const llm = new cha({
//   modelName: "claude-3-5-sonnet-20241022",
//   temperature: 0,
// });

const supervisorChain = formattedPrompt
    .pipe(
        llm.bindTools([routingTool], {
            tool_choice: 'route',
        })
    )
    // select the first one
    .pipe((x) => x.tool_calls?.[0]?.args ?? {});

// await supervisorChain.invoke({
//     messages: [
//         new HumanMessage({
//             content: 'write a report on birds.',
//         }),
//     ],
// });

import { START, StateGraph } from '@langchain/langgraph';

// 1. Create the graph
const workflow = new StateGraph(AgentState)
    // 2. Add the nodes; these will do the work
    .addNode('supported_queries', supportedQueriesAgentNode)
    .addNode('query_param_generator', queryParamGeneratorAgentNode)
    .addNode('fetch_users', fetchUsersAgentNode)
    .addNode('supervisor', supervisorChain);
// 3. Define the edges. We will define both regular and conditional ones
// After a worker completes, report to supervisor
members.forEach((member) => {
    workflow.addEdge(member, 'supervisor');
});

workflow.addConditionalEdges(
    'supervisor',
    (x: typeof AgentState.State) => x.next
);

workflow.addEdge(START, 'supervisor');

// const graph = workflow.compile();
export const graph = workflow.compile();


export const supervisorGithubAgent = async (message: string) => {
    const streamResults = graph.stream(
        {
            messages: [
                new HumanMessage({
                    content: message,
                }),
            ],
        },
        { recursionLimit: 100 }
    );

    for await (const output of await streamResults) {
        if (!output?.__end__) {
            console.log(output);
            console.log('----');
        }
    }

    const result = await graph.invoke({
        messages: [
            new HumanMessage({
                content: 'Give five react js user form new york?',
            }),
        ],
    });

    return result;
};
