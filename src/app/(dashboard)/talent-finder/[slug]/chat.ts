
import { Anthropic } from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { ChatOpenAI } from "@langchain/openai";
// import { getSystemPrompt, HTML_EXAMPLES } from '@/utils/promptExamples';
import { PromptTemplate } from "@langchain/core/prompts";
import { JsonOutputParser } from "@langchain/core/output_parsers";

export interface ChatResponse {
    responsetype: "text" | "html";
    response: string;
  }
  
export const sendMessage = async (
    message: string,
  ): Promise<ChatResponse> => {
    console.log("Sending message:", message);
  
    // const anthropicKey = localStorage.getItem('ANTHROPIC_API_KEY');
    // const r1Key = localStorage.getItem('R1DEEPSEEK_API_KEY');
  
    // if (!anthropicKey) {
    //   return {
    //     responsetype: "text",
    //     response: "Please provide the Anthropic API Key",
    //   };
    // }
  
    // // Initialize Anthropic client for potential formatting needs
    // const anthropic = new Anthropic({
    //   apiKey: anthropicKey,
    //   dangerouslyAllowBrowser: true,
    // });
  

    // console.log("Using Anthropic API");
    // console.log("env", process.env.NEXT_PUBLIC_OPENAI_API_KEY);
    try {
      const chatModel = new ChatOpenAI({
        // apiKey: "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",
        apiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY,
        temperature: 0,
        modelName: "gpt-4o-mini", // Use GPT-4 or any other model of your choice
      });
      const formatInstructions = `You are an expert front-end developer creating visually appealing, context-aware HTML responses with modern, responsive CSS styling using tailwindcss text color must balck. Your responses should be interactive, use smooth animations, hover effects, ui must small in size and incorporate relevant icons. Your responses should be tailored for a black and white theme and be ready for direct insertion into a webpage's body content. Only return the content of the body (such as a <div>), script, style and function calling. 
  
      Here are the expectations:
      
      - Prioritize modern, user-friendly designs with a focus on responsiveness.
      - Use smooth animations and transitions, including hover effects on buttons, cards, and icons.
      - Incorporate dynamic components such as buttons, icons, and modals.
      - Use icons (e.g., FontAwesome, Material Icons) where appropriate.
      - Ensure that all content is interactive and visually engaging.
      - Make use of a dark theme, using appropriate contrasting colors.
      - Use semantic HTML5 tags where appropriate.
      - Ensure that the code is compatible with modern browsers and devices.
      - If an image URL is provided, include the image in the output, ensuring it renders correctly within the design.
      - Ensure don't add view more button in the response.
      - Ensure link must open in new tab.
      
      
      Always respond in JSON format as follows:
      
      {{
        "responsetype": "html",
        "response": "<response_content>"
    }}
      
      Question: {question}
      `;
    //   const formatInstructions = `You are an expert front-end developer creating visually appealing, context-aware HTML responses with modern, responsive CSS styling. Your responses should be interactive, use smooth animations, hover effects, ui must small in size and incorporate relevant icons. Your responses should be tailored for a dark theme and be ready for direct insertion into a webpage's body content. Only return the content of the body (such as a <div>), script, style and function calling. 
  
    //   Here are the expectations:
      
    //   - Prioritize modern, user-friendly designs with a focus on responsiveness.
    //   - Use smooth animations and transitions, including hover effects on buttons, cards, and icons.
    //   - Incorporate dynamic components such as buttons, icons, and modals.
    //   - Use icons (e.g., FontAwesome, Material Icons) where appropriate.
    //   - Ensure that all content is interactive and visually engaging.
    //   - Make use of a dark theme, using appropriate contrasting colors.
    //   - Use semantic HTML5 tags where appropriate.
    //   - Ensure that the code is compatible with modern browsers and devices.
    //   - If an image URL is provided, include the image in the output, ensuring it renders correctly within the design.
    //   - Ensure don't add view more button in the response.
    //   - Ensure link must open in new tab.
      
      
    //   Always respond in JSON format as follows:
      
    //   {{
    //     "responsetype": "html",
    //     "response": "<response_content>"
    // }}
      
    //   Question: {question}
    //   `;
      try {
        const systemPrompt = new PromptTemplate({
          inputVariables: ["question"],
          template: formatInstructions,
        });
  
        // Create the chain to send the message and get a response
        const chain = systemPrompt.pipe(chatModel);
  
        // Invoke the chain with just the message (no history)
        const chatResponse = await chain.invoke({
          question: message,
          // openaikey: "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" // Pass the message
        });
  
        const content = chatResponse?.content;
        console.log("Raw response content:", content);
  
        // Use the JsonOutputParser to parse the response
        const parser = new JsonOutputParser<ChatResponse>();
  
        // Parse the content as JSON
        const parsedResponse = await parser.parse(content.toString());
        console.log("Parsed JSON response:", parsedResponse);
  
        return parsedResponse;
  
  
      } catch (error) {
        console.error("Error sending message with OpenAI:", error);
        return {
          responsetype: "text",
          response: "Sorry, there was an error processing your request.",
        };
      }
  
    } catch (error) {
      console.error("Error sending message:", error);
      return {
        responsetype: "text",
        response: "Sorry, there was an error processing your request.",
      };
    }
  };