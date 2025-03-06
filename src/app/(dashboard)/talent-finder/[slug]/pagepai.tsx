
// "use client"
// import { Button } from "@/components/ui/button"
// import { Input } from "@/components/ui/input"
// import React, { useState, useEffect, useRef } from 'react'
// import Image from "next/image"
// import agent from "@/image/ai-agent.gif"
// import github from "@/image/Github.gif"
// import dot from "@/image/dot.gif"
// import { ChevronUp, ChevronDown } from 'lucide-react';
// import AgentActivitySheet from "./AgentActivitySheet"
// import AgentSheet from "./AgentSheet"
// import { Checkbox } from '@/components/ui/checkbox';
// import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
// import { Label } from '@/components/ui/label';
// import { useSearchParams } from 'next/navigation'
// import { v4 as uuidv4 } from 'uuid';
// import { readStreamableValue } from "ai/rsc";
// import { runAgent } from "./action";
// import { StreamEvent } from "@langchain/core/tracers/log_stream";

// export default function Slug() {

//     const searchParams = useSearchParams()

//     const search = searchParams.get('query')

//     console.log("searchParams: " + (search ? search.replace("+", " ") : ""))

//     //   useEffect(() => {
//     //     // console.log("Agent Demo", createLangGraphAgent().then(res => console.log(res)));
//     //     fetch("/api/agent")
//     //       .then((res) => {
//     //         if (!res.ok) {
//     //           throw new Error("Network response was not ok");
//     //         }
//     //         return res.json();
//     //       })
//     //       .then((data) => {
//     //         setData(data.result);
//     //         setLoading(false);
//     //       })
//     //       .catch((err) => {
//     //         console.error("Error fetching agent output:", err);
//     //         setError(err.message);
//     //         setLoading(false);
//     //       });
//     //   }, []);

//     // const [data, setData] = useState<any>(null);
//     const [loading, setLoading] = useState<boolean>(false);
//     const [error, setError] = useState<string | null>(null);


//     const scrollRef = useRef<HTMLDivElement>(null);

//     //   console.log("data111111", data);

//     // console.log("data", data);

//     const aiResponse = async (newMessage: any) => {
//         fetch("/api/agent", { method: "POST", body: JSON.stringify(newMessage?.user) })
//             .then((res) => {
//                 if (!res.ok) {
//                     throw new Error("Network response was not ok");
//                 }
//                 return res.json();
//             })
//             .then((messages) => {
//                 console.log("messages", messages?.result?.messages)
//                 console.log("messages[messages.length - 1]?.kwargs?.content", messages?.result?.messages[messages?.result?.messages.length - 1]?.kwargs?.content)
//                 const updatedMessage = {
//                     id: uuidv4(),
//                     user: newMessage?.user,
//                     ai: messages?.result?.messages[messages?.result?.messages.length - 1]?.kwargs?.content
//                 };
//                 setChat((prevChat) => {
//                     // const aimsg = prevChat.find(chat => chat.id === newMessage.id)
//                     return prevChat.map((chat) =>
//                         chat?.id === newMessage.id ? { ...chat, ...updatedMessage } : chat
//                     );
//                 });
//                 // setChat((prevChat) => [...prevChat, newMessage]);
//                 // setData(messages.result);
//                 setLoading(false);
//             })
//             .catch((err) => {
//                 console.error("Error fetching agent output:", err);
//                 setError(err.message);
//                 setLoading(false);
//             });
//     }


//     const [selectedOptions, setSelectedOptions] = useState({
//         github: true,
//         ats: false,
//         reddit: false
//     });
//     const [submitted, setSubmitted] = useState(false);

//     const handleCheckboxChange = (value: keyof typeof selectedOptions) => {
//         setSelectedOptions(prev => ({
//             ...prev,
//             [value]: !prev[value]
//         }));
//     };


//     // Get the list of selected options for display
//     const getSelectedItems = () => {
//         return (Object.keys(selectedOptions) as (keyof typeof selectedOptions)[]).filter(key => selectedOptions[key]);
//     };

//     console.log("submitted", submitted)
//     const SelectPlatform = () => (<div className={`flex gap-5 justify-start ${submitted && "group opacity-50 pointer-events-none"}`}>
//         <div className="w-[60%] bg-muted p-2">
//             <div className="pb-2">Hey!</div>
//             <hr />
//             <div>
//                 <Card className="w-full max-w-md">
//                     <CardHeader>
//                         <CardTitle>Platform Selection</CardTitle>
//                     </CardHeader>
//                     <CardContent className="space-y-4">
//                         <div className="flex items-center space-x-2">
//                             <Checkbox disabled={submitted}
//                                 id="github"
//                                 checked={selectedOptions.github}
//                                 onCheckedChange={() => handleCheckboxChange('github')}
//                             />
//                             <Label htmlFor="github">GitHub</Label>
//                         </div>

//                         <div className="flex items-center space-x-2">
//                             <Checkbox disabled={submitted}
//                                 id="ats"
//                                 checked={selectedOptions.ats}
//                                 onCheckedChange={() => handleCheckboxChange('ats')}
//                             />
//                             <Label htmlFor="ats">ATS</Label>
//                         </div>

//                         <div className="flex items-center space-x-2">
//                             <Checkbox disabled={submitted}
//                                 id="reddit"
//                                 checked={selectedOptions.reddit}
//                                 onCheckedChange={() => handleCheckboxChange('reddit')}
//                             />
//                             <Label htmlFor="reddit">Reddit</Label>
//                         </div>

//                         {submitted && getSelectedItems().length > 0 && (
//                             <div className="mt-4 p-4 rounded bg-slate-100">
//                                 <p>Selected platforms:</p>
//                                 <ul className="list-disc pl-5 mt-2">
//                                     {getSelectedItems().map(item => (
//                                         <li key={item} className="capitalize">{item}</li>
//                                     ))}
//                                 </ul>
//                             </div>
//                         )}

//                         {submitted && getSelectedItems().length === 0 && (
//                             <div className="mt-4 p-4 rounded bg-amber-100 text-amber-800">
//                                 <p>No platforms selected!</p>
//                             </div>
//                         )}
//                     </CardContent>
//                     {/* <CardFooter>
//             <Button
//               onClick={handleAccept}
//               className="w-full"
//             >
//               {submitted ? " Accepted" : "Accept"}
//             </Button>
//           </CardFooter> */}
//                 </Card>
//             </div>
//         </div>
//     </div>)


//     const messages = [
//         {
//             id: "12121",
//             user: (search ? search.replace("+", " ") : ""),
//             ai: <SelectPlatform />
//         },
//         // {
//         //   user: "Lorem ipsum dolor, sit amet consectetur adipisicing elit. Quod ut nisi ex, fuga saepe vero tenetur veniam aliquam at suscipit sapiente cupiditate? Esse maiores omnis necessitatibus tenetur magnam doloremque? Error.",
//         //   ai: "Lorem ipsum dolor, sit amet consectetur adipisicing elit. Quod ut nisi ex, fuga saepe vero tenetur veniam aliquam at suscipit sapiente cupiditate? Esse maiores omnis necessitatibus tenetur magnam doloremque? Error."
//         // },
//         // {
//         //   user: "Lorem ipsum dolor, sit amet consectetur adipisicing elit. Quod ut nisi ex, fuga saepe vero tenetur veniam aliquam at suscipit sapiente cupiditate? Esse maiores omnis necessitatibus tenetur magnam doloremque? Error.",
//         //   ai: "Lorem ipsum dolor, sit amet consectetur adipisicing elit. Quod ut nisi ex, fuga saepe vero tenetur veniam aliquam at suscipit sapiente cupiditate? Esse maiores omnis necessitatibus tenetur magnam doloremque? Error."
//         // }
//     ]
//     const [chat, setChat] = useState<{ id: string; user: string; ai: string | React.ReactNode }[]>(messages)
//     const [inputValue, setInputValue] = useState("")
//     const [isLoading, setIsLoading] = useState(false);
//     const [loadingIndex, setLoadingIndex] = useState(0);
//     const [expandMessage, setExpandMessage] = useState(true);


//     useEffect(() => {
//         if (scrollRef.current) {
//             scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
//         }
//     }, [chat]);
//     //   "Message sending to agent...",
//     //   "Thinking...",
//     //   "Recieve your message...",
//     //   "Github agent invoked...",
//     //   "Processing...",
//     //   "Response generated...",
//     //   "Finishing..."
//     // ];

//     const loadingMessages = [
//         { message: "Message sending to agent...", agent: "agent" },
//         { message: "Thinking...", agent: "agent" },
//         { message: "Recieve your message...", agent: "agent" },
//         { message: "Invoking github agent", agent: "agent" },
//         { message: "Github agent invoke", agent: "github-agent" },
//         { message: "Processing...", agent: "github-agent" },
//         { message: "Response generated...", agent: "github-agent" },
//         { message: "Finishing...", agent: "github-agent" },
//         { message: "Final Answer generated", agent: "agent" },
//         { message: "Finished...", agent: "agent" },

//     ];


//     const [displayedMessages, setDisplayedMessages] = useState<{ message: string; agent: string }[]>([]);

//     useEffect(() => {
//         let interval: ReturnType<typeof setInterval>;
//         // if (isLoading) {
//         //   interval = setInterval(() => {
//         //     setLoadingIndex((prev) => (prev + 1) % loadingMessages.length);
//         //   }, 2000); // Change every 2 seconds
//         // } else {
//         //   setLoadingIndex(0); // Reset loading message when done
//         // }
//         if (isLoading) {
//             interval = setInterval(() => {
//                 setLoadingIndex((prev) => {
//                     if (prev < loadingMessages.length) {
//                         setDisplayedMessages((msgs) => {
//                             // Ensure the message is only added once
//                             if (!msgs.includes(loadingMessages[prev])) {
//                                 return [...msgs, loadingMessages[prev]];
//                             }
//                             return msgs;
//                         });
//                         return prev + 1; // Increment without cycling
//                     }
//                     return prev;
//                 });
//             }, 2000); // Change every 2 seconds
//         } else {
//             setLoadingIndex(0); // Reset loading message when done
//             setDisplayedMessages([]); // Clear displayed messages
//         }
//         return () => clearInterval(interval);
//         //eslint-disable-next-line
//     }, [isLoading]);

//     // const addMessage = (userMessage: string) => {
//     //   const newMessage = {
//     //     user: userMessage,
//     //     ai: "Lorem ipsum dolor, sit amet consectetur adipisicing elit. Quod ut nisi ex, fuga saepe vero tenetur veniam aliquam at suscipit sapiente cupiditate? Esse maiores omnis necessitatibus tenetur magnam doloremque? Error."
//     //   };

//     //   setChat((prevChat) => [...prevChat, newMessage]);
//     // };

//       const [data, setData] = useState<StreamEvent[]>([]);


//     const handleAccept = async() => {
//         setSubmitted(true);

//         // const updatedMessage = {
//         //     id: uuidv4(),
//         //     user: chat[chat.length - 1]?.user,
//         //     ai: <div className="loader" />
//         // };

//         // const lastMessage = chat[chat.length - 1] = updatedMessage

//         setChat(
//             (prevMessages) => {
//                 // Clone the array
//                 const updatedMessages = [...prevMessages];
//                 // Update the text of the last message
//                 if (updatedMessages.length > 0) {
//                     updatedMessages[updatedMessages.length - 1] = {
//                         ...updatedMessages[updatedMessages.length - 1],
//                         user: chat[chat.length - 1]?.user,
//                         ai: <div className="loader" />
//                     };
//                 }

//                 aiResponse(updatedMessages[updatedMessages.length - 1])

//                 return updatedMessages;
//             }
//         );

//         const { streamData } = await runAgent(chat[chat.length - 1]?.user);
//         for await (const item of readStreamableValue(streamData)) {
//             setData((prev) => [...prev, item]);
//         }
//     };

//     const addMessage = (userMessage: string) => {
//         // setIsLoading(true); // Start loading when user sends message

//         if (!submitted) {
//             const newMessage = {
//                 id: uuidv4(),
//                 user: userMessage,
//                 ai: <SelectPlatform />, // Placeholder for AI response
//             };

//             setChat((prevChat) => [...prevChat, newMessage]);
//         } else {
//             const id = uuidv4()

//             const newMessage = {
//                 id,
//                 user: userMessage,
//                 ai: <div className="loader" />
//             };
//             setChat((prevChat) => [...prevChat, newMessage]);
//             setLoading(true)
//             aiResponse(newMessage)
//         }

//         // Simulate AI response delay
//         // setTimeout(() => {
//         //   setChat((prevChat) =>
//         //     prevChat.map((msg, index) =>
//         //       index === prevChat.length - 1
//         //         // ? { ...msg, ai: (!submitted ? <SelectPlatform/> : "AI response generated...") }
//         //         ? { ...msg, ai: (!submitted ? <SelectPlatform /> : "Hey! I'll review the data and identify candidates.") }
//         //         : msg
//         //     )
//         //   );
//         //   setIsLoading(false); // Stop loading
//         // }, 20000); // Simulated delay (6 seconds)
//     };

//     const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
//         e.preventDefault();
//         if (inputValue.trim()) {
//             addMessage(inputValue);
//             setInputValue(""); // Clear input after submission
//         }
//     };

//     // console.log("expandMessage", expandMessage)
//     // console.log("loadingMessages[loadingIndex].message", loadingMessages[loadingIndex].message)

//     return (
//         <div className="flex flex-col  overflow-hidden ">
//             <header className="flex shrink-0 items-center gap-2 border-b pb-2">
//                 <div className="flex justify-between items-center gap-2 px-3 w-[100%]">
//                     <div className="text-2xl font-bold">Welcome to skillsync</div>
//                     <AgentSheet displayedMessages={displayedMessages} />
//                 </div>
//             </header>
//             {/* <div className="ml-auto mx-5 mb-2">
//         <AgentSheet displayedMessages={displayedMessages} />
//       </div> */}
//             <div className="flex flex-col gap-8 h-[calc(100dvh-140px)] overflow-y-scroll">
//                 {chat?.map((res, index) => <> <div className="flex gap-5 justify-end" key={index}>
//                     <div className="max-w-[60%] p-5">
//                         {res.user}
//                     </div>
//                     <div className=" flex justify-center items-center w-[50px] h-[50px] bg-muted rounded-full ">
//                         US
//                     </div>
//                 </div>
//                     <div className="flex gap-5 justify-start">
//                         <div className=" flex justify-center items-center w-[50px] h-[50px] bg-[#000] text-white rounded-full">
//                             AI
//                         </div>
//                         <div className="w-[60%] bg-muted p-4">
//                             {/* {!res.ai && !submitted && <Image src={dot} alt="" className="rounded-full" />} */}
//                             {res.ai}
//                             {!submitted && typeof res.ai !== "string" && <div className="flex items-center gap-2">
//                                 <Button className="w-[200px]"
//                                     onClick={handleAccept} disabled={submitted}
//                                 >
//                                     {submitted ? " Done" : "Accept"}
//                                 </Button>
//                             </div>}
//                         </div>
//                     </div>
//                     {/* {index === 3 && <div className="flex gap-5 justify-start">
//             <div className=" flex justify-center items-center w-[50px] h-[50px] bg-[#000] text-white rounded-full">
//               AI
//             </div>
//             <div className="w-[60%] bg-muted p-4">
//               <div className="pb-2">Hey!</div>
//               <hr />
//               <div>
//                 <Card className="w-full max-w-md">
//                   <CardHeader>
//                     <CardTitle>Platform Selection</CardTitle>
//                   </CardHeader>
//                   <CardContent className="space-y-4">
//                     <div className="flex items-center space-x-2">
//                       <Checkbox
//                         id="github"
//                         checked={selectedOptions.github}
//                         onCheckedChange={() => handleCheckboxChange('github')}
//                       />
//                       <Label htmlFor="github">GitHub</Label>
//                     </div>

//                     <div className="flex items-center space-x-2">
//                       <Checkbox
//                         id="ats"
//                         checked={selectedOptions.ats}
//                         onCheckedChange={() => handleCheckboxChange('ats')}
//                       />
//                       <Label htmlFor="ats">ATS</Label>
//                     </div>

//                     <div className="flex items-center space-x-2">
//                       <Checkbox
//                         id="reddit"
//                         checked={selectedOptions.reddit}
//                         onCheckedChange={() => handleCheckboxChange('reddit')}
//                       />
//                       <Label htmlFor="reddit">Reddit</Label>
//                     </div>

//                     {submitted && getSelectedItems().length > 0 && (
//                       <div className="mt-4 p-4 rounded bg-slate-100">
//                         <p>Selected platforms:</p>
//                         <ul className="list-disc pl-5 mt-2">
//                           {getSelectedItems().map(item => (
//                             <li key={item} className="capitalize">{item}</li>
//                           ))}
//                         </ul>
//                       </div>
//                     )}

//                     {submitted && getSelectedItems().length === 0 && (
//                       <div className="mt-4 p-4 rounded bg-amber-100 text-amber-800">
//                         <p>No platforms selected!</p>
//                       </div>
//                     )}
//                   </CardContent>
//                   <CardFooter>
//                     <Button
//                       onClick={handleAccept}
//                       className="w-full"
//                     >
//                       Accept
//                     </Button>
//                   </CardFooter>
//                 </Card>
//               </div>
//             </div>
//           </div>} */}
//                 </>)}
//             </div>
//             <div className="flex justify-center mt-auto w-[100%] p-2">
//                 <div className="mt-auto grow">
//                     {isLoading && submitted && <div className={`absolute bottom-[89px] bg-muted w-[95%] flex justify-between   gap-5 p-2 rounded-t-5`}>
//                         {expandMessage && <div className="flex flex-col gap-2">
//                             <div className="flex items-center gap-2">
//                                 <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
//                                     <Image src={agent} alt="" className="rounded-full" />
//                                 </div>
//                                 <div className="">
//                                     <p className="text-gray-500 text-md">Supervisor agent</p>
//                                     <p className="text-gray-500 text-sm">{loadingMessages[loadingIndex]?.agent === "agent" && loadingMessages[loadingIndex].message}</p>
//                                 </div>
//                             </div>
//                             <div className="flex items-center gap-2">
//                                 <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
//                                     <Image src={github} alt="" className="rounded-full" />
//                                 </div>
//                                 <div className="">
//                                     <p className="text-gray-500 text-md">Github</p>
//                                     <p className="text-gray-500 text-sm">{loadingMessages[loadingIndex]?.agent === "github-agent" && loadingMessages[loadingIndex].message}</p>
//                                 </div>
//                             </div>
//                         </div>}
//                         {!expandMessage && submitted && <div>
//                             {loadingMessages[loadingIndex]?.agent === "agent" && <div className="flex items-center gap-2">
//                                 <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
//                                     <Image src={agent} alt="" className="rounded-full" />
//                                 </div>
//                                 <div className="">
//                                     <p className="text-gray-500 text-md">Supervisor agent</p>
//                                     <p className="text-gray-500 text-sm">{loadingMessages[loadingIndex]?.agent === "agent" && loadingMessages[loadingIndex].message}</p>
//                                 </div>
//                             </div>}
//                             {loadingMessages[loadingIndex]?.agent === "github-agent" && <div className="flex items-center gap-2">
//                                 <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
//                                     <Image src={github} alt="" className="rounded-full" />
//                                 </div>
//                                 <div className="">
//                                     <p className="text-gray-500 text-md">Github</p>
//                                     <p className="text-gray-500 text-sm">{loadingMessages[loadingIndex]?.agent === "github-agent" && loadingMessages[loadingIndex].message}</p>
//                                 </div>
//                             </div>}
//                         </div>
//                         }
//                         <div>
//                             <div className="flex gap-[10]"
//                                 onClick={() => setExpandMessage((prev) => !prev)}>
//                                 <AgentActivitySheet />
//                                 <div className="bg-white rounded-full p-2 cursor-pointer" >
//                                     {expandMessage ? <ChevronDown /> : <ChevronUp />}
//                                 </div>
//                             </div>
//                         </div>
//                     </div>
//                     }
//                     <div className="flex justify-center items-center gap-4 w-full">
//                         <form onSubmit={handleSubmit} className="flex justify-center items-center gap-4 w-full">
//                             <Input
//                                 className="h-[50px]"
//                                 value={inputValue}
//                                 onChange={(e) => setInputValue(e.target.value)}
//                             />
//                             <Button type="submit" size="sm" className="px-3">
//                                 Submit
//                             </Button>
//                         </form>
//                     </div>
//                     {/* <AgentActivityDrawer/> */}
//                 </div>
//             </div>
//         </div>
//     )
// }


// // 'use client'

// // import { useChat } from 'ai/react'
// // import Link from 'next/link'

// // export default function Page() {
// //   const { messages, input, handleInputChange, handleSubmit, isLoading } =
// //     useChat({ experimental_throttle: 50, api: '/api/agent' })

// //   return (
// //     <div className="flex flex-col container mx-auto p-6 w-dvw h-dvh gap-4 items-start justify-between">
// //       <Link href="/" className="bg-blue-500 text-white p-2 rounded-md">
// //         Back to Home
// //       </Link>
// //       <div className="w-full flex-1">
// //         {messages.map((message) => (
// //           <div
// //             key={message.id}
// //             className={`p-4 ${
// //               message.role === 'user' ? 'bg-gray-100' : 'bg-gray-200'
// //             }`}
// //           >
// //             {message.role === 'user' ? 'User: ' : 'AI: '}
// //             {message.content}
// //           </div>
// //         ))}
// //         {isLoading && (
// //           <div className="flex flex-col p-2 gap-2 justify-center items-center">
// //             <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
// //             <button
// //               type="button"
// //               onClick={() => stop()}
// //               className="bg-red-500 text-white p-2 rounded-md"
// //             >
// //               Stop
// //             </button>
// //           </div>
// //         )}
// //       </div>

// //       <form onSubmit={handleSubmit} className="w-full flex gap-2">
// //         <input
// //           name="prompt"
// //           className="w-full p-2 rounded-md border border-blue-500"
// //           autoFocus
// //           disabled={isLoading}
// //           value={input}
// //           onChange={handleInputChange}
// //         />
// //         <button type="submit" className="bg-blue-500 text-white p-2 rounded-md">
// //           Send
// //         </button>
// //       </form>
// //     </div>
// //   )
// // }
// 2