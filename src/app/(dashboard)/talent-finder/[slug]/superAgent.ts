import { END, START, StateGraph } from "@langchain/langgraph";
import { RunnableLambda } from "@langchain/core/runnables";
import { Annotation } from "@langchain/langgraph";
import { HumanMessage, BaseMessage, SystemMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { z } from "zod";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { JsonOutputToolsParser } from "langchain/output_parsers";
import { ChatOpenAI } from "@langchain/openai";
import { Runnable } from "@langchain/core/runnables";
import { StructuredToolInterface } from "@langchain/core/tools";
import { MessagesAnnotation } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";

const llm = new ChatOpenAI({ modelName: "gpt-4o-mini" });

const agentStateModifier = (
  systemPrompt: string,
  tools: StructuredToolInterface[],
  teamMembers: string[],
): ((state: typeof MessagesAnnotation.State) => BaseMessage[]) => {
  const toolNames = tools.map((t) => t.name).join(", ");
  const systemMsgStart = new SystemMessage(systemPrompt +
    "\nWork autonomously according to your specialty, using the tools available to you." +
    " Do not ask for clarification." +
    " Your other team members (and other teams) will collaborate with you with their own specialties." +
    ` You are chosen for a reason! You are one of the following team members: ${teamMembers.join(", ")}.`)
  const systemMsgEnd = new SystemMessage(`Supervisor instructions: ${systemPrompt}\n` +
    `Remember, you individually can only use these tools: ${toolNames}` +
    "\n\nEnd if you have already completed the requested task. Communicate the work completed.");

  return (state: typeof MessagesAnnotation.State): any[] =>
    [systemMsgStart, ...state.messages, systemMsgEnd];
}

async function runAgentNode(params: {
  state: any;
  agent: Runnable;
  name: string;
}) {
  const { state, agent, name } = params;
  const result = await agent.invoke({
    messages: state.messages,
  });

  // console.log("result", result);
  const lastMessage = result.messages[result.messages.length - 1];
  // console.log("lastMessage", lastMessage);
  return {
    messages: [new HumanMessage({ content: lastMessage.content, name })],
  };
}

async function createTeamSupervisor(
  llm: ChatOpenAI,
  systemPrompt: string,
  members: string[],
): Promise<Runnable> {
  const options = ["FINISH", ...members];
  const routeTool = {
    name: "route",
    description: "Select the next role.",
    schema: z.object({
      reasoning: z.string(),
      next: z.enum(["FINISH", ...members]),
      instructions: z.string().describe("The specific instructions of the sub-task the next role should accomplish."),
    })
  }
  let prompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder("messages"),
    [
      "system",
      "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
    ],
  ]);
  prompt = await prompt.partial({
    options: options.join(", "),
    team_members: members.join(", "),
  });

  const supervisor = prompt
    .pipe(
      llm.bindTools([routeTool], {
        tool_choice: "route",
      }),
    )
    .pipe(new JsonOutputToolsParser())
    // select the first one
    .pipe((x) => ({
      next: x[0].args.next,
      instructions: x[0].args.instructions,
    }));

  return supervisor;
}

// Define the top-level State interface
const State = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  next: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "GithubTeam",
  }),
  instructions: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "Resolve the user's request.",
  }),
});

const GithubTeamState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  team_members: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
  }),
  next: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "supervisor",
  }),
  instructions: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "Solve the human's question.",
  }),
})


const supportedQueriesAgentNode = (state: typeof GithubTeamState.State) => {
  const stateModifier = agentStateModifier(
    `
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
    `,
    [],
    state.team_members ?? ["SupportedQueries"],
  )
  const supportedQueriesAgent = createReactAgent({
    llm,
    tools: [],
    stateModifier,
  })
  return runAgentNode({ state, agent: supportedQueriesAgent, name: "SupportedQueries" });
};

const queryParamGeneratorAgentNode = (state: typeof GithubTeamState.State) => {
  const stateModifier = agentStateModifier(
    `
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
    `,
    [],
    state.team_members ?? ["QueriesGenerator"],
  )
  const queryParamGeneratorAgent = createReactAgent({
    llm,
    tools: [],
    stateModifier,
  })
  return runAgentNode({ state, agent: queryParamGeneratorAgent, name: "QueriesGenerator" });
};

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
      console.log("Fetching users with query:", queryParam);
      perPage = Math.min(Math.max(perPage, 1), 100);
      const url = `https://api.github.com/search/users?q=${queryParam}&per_page=${perPage}`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(
          `GitHub API error: ${res.status} ${res.statusText}`
        );
      }
      const data = await res.json();
      console.log('GitHub API response:', data);
      return data
    } catch (error) {
      console.error('Error fetching users:', error);
      throw new Error(
        'Failed to fetch GitHub users. Please check your input and try again.'
      );
    }
  },
  {
    name: 'fetchUsers',
    description: 'Returns a list of GitHub users based on search query',
    schema: fetchUsersSchema, // Attach the Zod schema
  }
);

const fetchUsersAgentNode = (state: typeof GithubTeamState.State) => {
  const stateModifier = agentStateModifier(
    `
    You are a specialized Query Parameters Filter Agent. Your task is to analyze a Job Description written in natural language and extract only the relevant fields supported by the GitHub User Search API. Focus exclusively on fields that are searchable via the API.
    `,
    [fetchUsers],
    state.team_members ?? ["fetchUsers"],
  )
  const fetchUsersAgent = createReactAgent({
    llm,
    tools: [fetchUsers],
    stateModifier,
  })
  return runAgentNode({ state, agent: fetchUsersAgent, name: "fetchUsers" });
};

// const members = ['SupportedQueries', 'QueriesGenerator', 'fetchUsers']

const githubSupervisor = await createTeamSupervisor(
  llm,
  "You are a supervisor tasked with managing a conversation between the " +
  "following workers:  {team_members}. Given the following user request, " +
  "ALWAYS respond with SupportedQueries worker as it will be called first. " +
  "THEN , take its output as parameter to the QueriesGenerator worker " +
  "THEN , take the output of QueriesGenerator as parameter to fetchUsers worker " +
  "After calling fetchUsers , respond with FINISH.",
  ["SupportedQueries", "QueriesGenerator", "fetchUsers"],
);
// const githubSupervisor = await createTeamSupervisor(
//   llm,
//   "You are a supervisor tasked with managing a conversation between the" +
//   " following workers:  {team_members}. Given the following user request," +
//   " respond with the worker to act next. Each worker will perform a" +
//   " task and respond with their results and status. When finished," +
//   " respond with FINISH.\n\n" +
//   " Select strategically to minimize the number of steps taken.",
//   ['SupportedQueries', 'QueriesGenerator', 'fetchUsers'],
// );

// Create the graph here:
const githubGraph = new StateGraph(GithubTeamState)
  .addNode("SupportedQueries", supportedQueriesAgentNode)
  .addNode("QueriesGenerator", queryParamGeneratorAgentNode)
  .addNode("fetchUsers", fetchUsersAgentNode)
  .addNode("supervisor", githubSupervisor)
  // Add the edges that always occur
  .addEdge("SupportedQueries", "supervisor")
  .addEdge("QueriesGenerator", "supervisor")
  .addEdge("fetchUsers", "supervisor")
  // Add the edges where routing applies
  .addConditionalEdges("supervisor", (x) => x.next, {
    SupportedQueries: "SupportedQueries",
    QueriesGenerator: "QueriesGenerator",
    fetchUsers: "fetchUsers",
    FINISH: END,
  })
  .addEdge(START, "supervisor");

const enterAuthoringChainGithub = RunnableLambda.from(
  ({ messages }: { messages: BaseMessage[] }) => {
    return {
      messages: messages,
      team_members: ['SupportedQueries', 'QueriesGenerator', 'fetchUsers'],
    };
  },
);

// const githubChain = githubGraph.compile();

const authoringChainGithub = enterAuthoringChainGithub.pipe(githubGraph.compile());


const BooleanTeamState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  team_members: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
  }),
  next: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "supervisor",
  }),
  instructions: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "Solve the human's question.",
  }),
})

const createBooleanAgentNode = (state: typeof BooleanTeamState.State) => {
  const stateModifier = agentStateModifier(
    `
    You are an expert in crafting Boolean search queries for recruitment purposes. Your task is to generate a Boolean search query using the following job description. Ensure the query accurately reflects the required and preferred qualifications, skills, and experience mentioned. Use the correct Boolean operators (AND, OR, NOT, and parentheses) to group terms appropriately. If multiple skills or qualifications are listed, use AND for mandatory requirements and OR for optional ones. Ensure exact phrases (like "Bachelor's degree") are enclosed in quotes.
    `,
    [],
    state.team_members ?? ["createBooleanQueryAgent"],
  )
  const createBooleanQueryAgent = createReactAgent({
    llm,
    tools: [],
    stateModifier,
  })
  return runAgentNode({ state, agent: createBooleanQueryAgent, name: "createBooleanQueryAgent" });
};

// const members = ['SupportedQueries', 'QueriesGenerator', 'fetchUsers']

const booleanSupervisor = await createTeamSupervisor(
  llm,
  "You are a supervisor tasked with managing a conversation between the" +
  " following workers:  {team_members}. Given the following user request," +
  " respond with the worker to act next. Each worker will perform a" +
  " task and respond with their results and status. When finished," +
  " respond with FINISH.\n\n" +
  " Select strategically to minimize the number of steps taken.",
  ["createBooleanQueryAgent"],
);

// const githubSupervisor = await createTeamSupervisor(
//   llm,
//   "You are a supervisor tasked with managing a conversation between the" +
//   " following workers:  {team_members}. Given the following user request," +
//   " respond with the worker to act next. Each worker will perform a" +
//   " task and respond with their results and status. When finished," +
//   " respond with FINISH.\n\n" +
//   " Select strategically to minimize the number of steps taken.",
//   ['SupportedQueries', 'QueriesGenerator', 'fetchUsers'],
// );

// Create the graph here:
const booleanGraph = new StateGraph(BooleanTeamState)
  .addNode("createBooleanQueryAgent", createBooleanAgentNode)
  .addNode("supervisor", booleanSupervisor)
  // Add the edges that always occur
  .addEdge("createBooleanQueryAgent", "supervisor")
  // Add the edges where routing applies
  .addConditionalEdges("supervisor", (x) => x.next, {
    createBooleanQueryAgent: "createBooleanQueryAgent",
    FINISH: END,
  })
  .addEdge(START, "supervisor");

const enterAuthoringChainBoolean = RunnableLambda.from(
  ({ messages }: { messages: BaseMessage[] }) => {
    return {
      messages: messages,
      team_members: ['createBooleanQueryAgent'],
    };
  },
);

// const githubChain = githubGraph.compile();

const authoringChainBoolean = enterAuthoringChainBoolean.pipe(booleanGraph.compile());

const supervisorNode = await createTeamSupervisor(
  llm,
  // "You are a supervisor tasked with managing a conversation between the" +
  // " following teams: {team_members}. Given the following user request," +
  // " respond with the worker to act next. Each worker will perform a" +
  // " task and respond with their results and status. When finished," +
  // " respond with FINISH.\n\n" +
  // " Select strategically to minimize the number of steps taken.",
  "You are a supervisor tasked with managing a conversation between the" +
  " following teams: {team_members}. Given the following user request," +
  " select one worker to perform their task. Once any worker responds" +
  " with their results and status, immediately end the process by responding with FINISH.",
  ["GithubTeam", "BooleanTeam"],
);

const getMessages = RunnableLambda.from((state: typeof State.State) => {
  return { messages: state.messages };
});

const joinGraph = RunnableLambda.from((response: any) => {
  return {
    messages: [response.messages[response.messages.length - 1]],
  };
});

const superGraph = new StateGraph(State)
  .addNode("GithubTeam", getMessages.pipe(authoringChainGithub).pipe(joinGraph))
  .addNode("BooleanTeam", getMessages.pipe(authoringChainBoolean).pipe(joinGraph))
  .addNode("supervisor", supervisorNode)
  // .addEdge("ResearchTeam", "supervisor")
  // .addEdge("PaperWritingTeam", "supervisor")
  .addEdge("GithubTeam", "supervisor")
  .addEdge("BooleanTeam", "supervisor")
  .addConditionalEdges("supervisor", (x) => x.next, {
    // PaperWritingTeam: "PaperWritingTeam",
    // ResearchTeam: "ResearchTeam",
    GithubTeam: "GithubTeam",
    BooleanTeam: "BooleanTeam",
    FINISH: END,
  })
  .addEdge(START, "supervisor");


export const compiledSuperGraph = superGraph.compile();

// const resultStream = compiledSuperGraph.stream(
//   {
//     messages: [
//       new HumanMessage(
//         "Look up a current event, write a poem about it, then plot a bar chart of the distribution of words therein.",
//       ),
//     ],
//   },
//   { recursionLimit: 150 },
// );

// for await (const step of await resultStream) {
//   if (!step.__end__) {
//     console.log(step);
//     console.log("---");
//   }
// }