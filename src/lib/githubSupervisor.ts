// // Import dependencies
// // import * as githubAgent from './github_agent'; // Expected to export supportedQueriesNode, queryParamGeneratorNode, fetchUsersNode
// import { ChatOpenAI } from 'langchain/openai';
// import { createReactAgent } from 'langgraph/prebuilt';
// import { createTeamSupervisorFunc } from './utils/create_team_supervisor_func';
// import { StateGraph, START, END } from 'langgraph/graph';
// import { MemorySaver } from 'langgraph/checkpoint/memory';
// import { HumanMessage } from 'langchain/messages';

// // Initialize the memory checkpointer
// const memory = new MemorySaver();

// /**
//  * @typedef {Object} GithubTeamState
//  * @property {Array<import('langchain/messages').BaseMessage>} messages - A message is added after each team member finishes.
//  * @property {string[]} team_members - The team members are tracked so they are aware of others' skill-sets.
//  * @property {string} next - Used to route work. The supervisor updates this after each decision.
//  */

// /**
//  * Converts a plain text message into the initial chain state.
//  * @param {string} message - The initial user message.
//  * @returns {object} - The state with a messages array.
//  */
// function enterChain(message) {
//   return {
//     messages: [new HumanMessage({ content: message })],
//   };
// }

// /**
//  * Conditional edge to decide the next step.
//  * If the last message starts with "Error:" then it returns "query_param_generator" otherwise "supervisor".
//  * @param {GithubTeamState} state - The current state.
//  * @returns {"supervisor" | "query_param_generator"}
//  */
// function condition(state) {
//   const messages = state.messages;
//   console.log("messages inside condition conditional edge:", messages);
//   const lastMessage = messages[messages.length - 1];
//   console.log("last_message here in conditional edge:", lastMessage);
//   if (lastMessage.content.startsWith("Error:")) {
//     return "query_param_generator";
//   } else {
//     return "supervisor";
//   }
// }

// /**
//  * Creates the GitHub team supervisor chain.
//  * @param {Function} agentNode - A function to create agent nodes (expects an object with {agent, name}).
//  * @returns {Function} - A function that accepts an initial message and processes the chain.
//  */
// function githubTeamSupervisor(agentNode) {
//   // Initialize the language model
//   const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

//   // Retrieve the agent nodes from the githubAgent module
//   const supportedQueriesAgentNode = githubAgent.supportedQueriesNode(agentNode);
//   const queryParamNode = githubAgent.queryParamGeneratorNode(agentNode);
//   const fetchNode = githubAgent.fetchUsersNode(agentNode);

//   // Define the team-level supervisor agent
//   function supervisorAgent(state) {
//     const githubSupervisorAgent = createTeamSupervisorFunc(
//       llm,
//       "You are a supervisor tasked with managing a conversation between the following workers: supported_queries, query_param_generator, fetch_users. Given the following user request, ALWAYS respond with supported_queries worker as it will be called first. THEN, take its output as parameter to the query_param_generator worker. THEN, take the output of query_param_generator as parameter to fetch_users worker. After calling fetch_users, respond with FINISH.",
//       ["supported_queries", "query_param_generator", "fetch_users"]
//     );
//     return githubSupervisorAgent;
//   }

//   // Create a new state graph
//   const githubGraph = new StateGraph();

//   // Add team-level nodes
//   githubGraph.addNode("supported_queries", supportedQueriesAgentNode);
//   githubGraph.addNode("query_param_generator", queryParamNode);
//   githubGraph.addNode("fetch_users", fetchNode);
//   githubGraph.addNode("supervisor", supervisorAgent);

//   // Define the control flow edges
//   githubGraph.addEdge(START, "supported_queries");
//   githubGraph.addEdge("supported_queries", "query_param_generator");
//   githubGraph.addEdge("query_param_generator", "fetch_users");
//   githubGraph.addConditionalEdges("fetch_users", condition);
//   githubGraph.addEdge("supervisor", END);
//   // (Optional alternative edge: githubGraph.addEdge(START, "supervisor");)

//   // Compile the graph with the memory checkpointer
//   const chain = githubGraph.compile({ checkpointer: memory });

//   // Compose the final chain: first run enterChain then pass its output to the compiled graph.
//   const githubChain = (message) => chain(enterChain(message));
//   // Optionally, you can create a graph image if your utility is available:
//   // createImageFunc.createGraphImage(chain, "github_graph_image");

//   return githubChain;
// }

// export { githubTeamSupervisor };
