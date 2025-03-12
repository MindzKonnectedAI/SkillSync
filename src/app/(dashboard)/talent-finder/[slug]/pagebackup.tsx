"use client"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import React, { useState, useEffect, useRef } from 'react'
import Image from "next/image"
import agent from "@/image/ai-agent.gif"
// import github from "@/image/Github.gif"
// import dot from "@/image/dot.gif"
import { ChevronUp, ChevronDown } from 'lucide-react';
import AgentActivitySheet from "./AgentActivitySheet"
import AgentSheet from "./AgentSheet"
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { useSearchParams } from 'next/navigation'
import { v4 as uuidv4 } from 'uuid';
import { readStreamableValue } from "ai/rsc";
import { runAgent } from "./action";
import { StreamEvent } from "@langchain/core/tracers/log_stream";
import { ChatOpenAI } from '@langchain/openai';
import { sendMessage } from "./chat"

const AssistantResponse = ({ content }: { content: string }) => {
    useEffect(() => {
        // Create a temporary container
        const container = document.createElement('div');
        container.innerHTML = content;

        // Find all external scripts and inline scripts
        const scripts = Array.from(container.getElementsByTagName('script'));
        const externalScripts = scripts.filter(script => script.src);
        const inlineScripts = scripts.filter(script => !script.src);

        // Function to load external script
        const loadExternalScript = (script: HTMLScriptElement): Promise<void> => {
            return new Promise((resolve, reject) => {
                const newScript = document.createElement('script');

                // Copy all attributes
                Array.from(script.attributes).forEach(attr => {
                    newScript.setAttribute(attr.name, attr.value);
                });

                newScript.onload = () => resolve();
                newScript.onerror = () => reject();

                document.body.appendChild(newScript);
            });
        };

        // Function to execute inline script
        const executeInlineScript = (script: HTMLScriptElement) => {
            const newScript = document.createElement('script');

            // Copy all attributes
            Array.from(script.attributes).forEach(attr => {
                newScript.setAttribute(attr.name, attr.value);
            });

            // Execute script
            const scriptContent = script.innerHTML;
            newScript.innerHTML = `
          try {
            ${scriptContent}
          } catch (error) {
            console.error('Error executing script:', error);
          }
        `;

            document.body.appendChild(newScript);
            return newScript;
        };

        // Keep track of added scripts for cleanup
        const addedScripts: HTMLScriptElement[] = [];

        // Load all scripts in sequence
        const loadAllScripts = async () => {
            // First load all external scripts
            for (const script of externalScripts) {
                try {
                    await loadExternalScript(script);
                } catch (error) {
                    console.error('Error loading external script:', error);
                }
            }

            // Then execute inline scripts
            for (const script of inlineScripts) {
                const newScript = executeInlineScript(script);
                addedScripts.push(newScript);
            }
        };

        // Start loading scripts
        loadAllScripts();

        // Add the HTML content
        const contentDiv = document.createElement('div');
        contentDiv.innerHTML = content;
        // Remove script tags from the content to prevent double execution
        Array.from(contentDiv.getElementsByTagName('script')).forEach(script => script.remove());

        return () => {
            // Cleanup scripts on unmount
            addedScripts.forEach(script => {
                if (script && script.parentNode) {
                    script.parentNode.removeChild(script);
                }
            });
        };
    }, [content]);

    return <div dangerouslySetInnerHTML={{ __html: content }} />;
};


export default function Slug() {
    const searchParams = useSearchParams()
    const search = searchParams.get('query')
    const scrollRef = useRef<HTMLDivElement>(null);

    const [selectedOptions, setSelectedOptions] = useState({
        github: true,
        ats: false,
        reddit: false
    });
    const [submitted, setSubmitted] = useState(false);
    const [inputValue, setInputValue] = useState("")
    const [isLoading, setIsLoading] = useState(false);
    const [expandMessage, setExpandMessage] = useState(true);
    const [data, setData] = useState<StreamEvent[]>([]);
    const handleCheckboxChange = (value: keyof typeof selectedOptions) => {
        setSelectedOptions(prev => ({
            ...prev,
            [value]: !prev[value]
        }));
    };


    // Get the list of selected options for display
    const getSelectedItems = () => {
        return (Object.keys(selectedOptions) as (keyof typeof selectedOptions)[]).filter(key => selectedOptions[key]);
    };

    const SelectPlatform = () => (<div className={`flex gap-5 justify-start ${submitted && "group opacity-50 pointer-events-none"}`}>
        <div className="w-[60%] bg-muted p-2">
            <div className="pb-2">Hey!</div>
            <hr />
            <div>
                <Card className="w-full max-w-md">
                    <CardHeader>
                        <CardTitle>Platform Selection</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex items-center space-x-2">
                            <Checkbox disabled={submitted}
                                id="github"
                                checked={selectedOptions.github}
                                onCheckedChange={() => handleCheckboxChange('github')}
                            />
                            <Label htmlFor="github">GitHub</Label>
                        </div>

                        <div className="flex items-center space-x-2">
                            <Checkbox disabled={submitted}
                                id="ats"
                                checked={selectedOptions.ats}
                                onCheckedChange={() => handleCheckboxChange('ats')}
                            />
                            <Label htmlFor="ats">ATS</Label>
                        </div>

                        <div className="flex items-center space-x-2">
                            <Checkbox disabled={submitted}
                                id="reddit"
                                checked={selectedOptions.reddit}
                                onCheckedChange={() => handleCheckboxChange('reddit')}
                            />
                            <Label htmlFor="reddit">Reddit</Label>
                        </div>

                        {submitted && getSelectedItems().length > 0 && (
                            <div className="mt-4 p-4 rounded bg-slate-100">
                                <p>Selected platforms:</p>
                                <ul className="list-disc pl-5 mt-2">
                                    {getSelectedItems().map(item => (
                                        <li key={item} className="capitalize">{item}</li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        {submitted && getSelectedItems().length === 0 && (
                            <div className="mt-4 p-4 rounded bg-amber-100 text-amber-800">
                                <p>No platforms selected!</p>
                            </div>
                        )}
                    </CardContent>
                    <CardFooter>
                        <Button
                            onClick={handleAccept}
                            className="w-full"
                        >
                            {submitted ? " Accepted" : "Accept"}
                        </Button>
                    </CardFooter>
                </Card>
            </div>
        </div>
    </div>)


    const messages = [
        {
            id: "12121",
            user: (search ? search.replace("+", " ") : ""),
            ai: <SelectPlatform />
        },
    ]
    const [chat, setChat] = useState<{ id: string; user: string; ai: string | React.ReactNode }[]>(messages)

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [chat]);

    const streamFn = async (input: string) => {

        const arr = []
        const { streamData } = await runAgent(input);
        for await (const item of readStreamableValue(streamData)) {
            setData((prev) => [...prev, item]);
            arr.push(item)
        }
        return arr
    }

    type handleUpdateMessageStreamType = {
        id: string;
        user: string;
        ai: string | React.ReactNode;
    }

    const handleUpdateMessageStream = async (props: handleUpdateMessageStreamType) => {
        const questioWithTeam = `${props.user} "Team: ${"GithubTeam"}`

        const data = await streamFn(questioWithTeam)
        // console.log("arr:", data)
        // console.log("streamData:", streamData)
        const secondlastNode = data[data.length - 2]
        // console.log("secondlastNode", secondlastNode)

        const nodeName = Object.keys(secondlastNode)[0]
        // console.log("nodeName", nodeName)
        const lastContent = secondlastNode[nodeName]?.messages[0]?.kwargs?.content
        // console.log("lastContent", lastContent)

        // const history = newMessages.map(({ role, content }) => ({
        //     role,
        //     content: typeof content === "string" ? content : "",
        // }));

        const response = await sendMessage(lastContent);

        // const messages = [
        //     new HumanMessage(
        //       `Create next js ui component for give context and very important only return conponent nothing else: ${lastContent}`
        //     ),
        //   ];

        // const response = await chatModel.invoke(messages);

        console.log("response", response)

        const updatedMessage = {
            id: uuidv4(),
            user: props.user,
            ai: response.responsetype === "html"
                ? <AssistantResponse content={response.response} />
                : response.response
        };

        setChat((prevChat) => {
            return prevChat.map((res) =>
                res?.id === props.id ? { ...res, ...updatedMessage } : res
            );
        });
        setIsLoading(false); // Start loading when user sends message

    }

    const handleAccept = async () => {
        setSubmitted(true);
        setIsLoading(true);
        setChat(
            (prevMessages) => {
                // Clone the array
                const updatedMessages = [...prevMessages];
                // Update the text of the last message
                if (updatedMessages.length > 0) {
                    updatedMessages[updatedMessages.length - 1] = {
                        ...updatedMessages[updatedMessages.length - 1],
                        user: chat[chat.length - 1]?.user,
                        ai: <div className="loader" />
                    };
                }
                return updatedMessages;
            }
        );

        await handleUpdateMessageStream(chat[chat.length - 1])

        // const questioWithTeam = `${chat[chat.length - 1]?.user} "Team: ${"GithubTeam"}`

        // const data = await streamFn(questioWithTeam)
        // console.log("arr:", data)
        // // console.log("streamData:", streamData)
        // const secondlastNode = data[data.length - 2]
        // console.log("secondlastNode", secondlastNode)

        // const nodeName = Object.keys(secondlastNode)[0]
        // console.log("nodeName", nodeName)
        // const lastContent = secondlastNode[nodeName]?.messages[0]?.kwargs?.content
        // console.log("lastContent", lastContent)

        // const updatedMessage = {
        //     id: uuidv4(),
        //     user: chat[chat.length - 1]?.user,
        //     ai: lastContent
        // };

        // setChat((prevChat) => {
        //     return prevChat.map((res) =>
        //         res?.id === chat[chat.length - 1].id ? { ...res, ...updatedMessage } : res
        //     );
        // });

        setExpandMessage(false)
    };



    const addMessage = async (userMessage: string) => {
        setIsLoading(true); // Start loading when user sends message

        if (!submitted) {
            const newMessage = {
                id: uuidv4(),
                user: userMessage,
                ai: <SelectPlatform />, // Placeholder for AI response
            };

            setChat((prevChat) => [...prevChat, newMessage]);
        } else {
            const id = uuidv4()

            const newMessage = {
                id,
                user: userMessage,
                ai: <div className="loader" />
            };
            setChat((prevChat) => [...prevChat, newMessage]);
            // aiResponse(newMessage)

            await handleUpdateMessageStream(newMessage)

            // const questioWithTeam = `${userMessage} "Team: ${"GithubTeam"}`

            // const data = await streamFn(questioWithTeam)
            // console.log("arr:", data)
            // // console.log("streamData:", streamData)
            // const secondlastNode = data[data.length - 2]
            // console.log("secondlastNode", secondlastNode)

            // const nodeName = Object?.keys(secondlastNode)[0]
            // console.log("nodeName", nodeName)
            // const lastContent = secondlastNode[nodeName]?.messages[0]?.kwargs?.content
            // console.log("lastContent", lastContent)

            // const updatedMessage = {
            //     id: uuidv4(),
            //     user: chat[chat.length - 1]?.user,
            //     ai: lastContent
            // };

            // setChat((prevChat) => {
            //     return prevChat.map((res) =>
            //         res?.id === id ? { ...res, ...updatedMessage } : res
            //     );
            // });

        }
    };

    // console.log("dataStream:", data)

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setData([]);
        if (inputValue.trim()) {
            addMessage(inputValue);
            setInputValue(""); // Clear input after submission
        }
    };


    return (
        <div className="flex flex-col  overflow-hidden ">
            <header className="flex shrink-0 items-center gap-2 border-b pb-2">
                <div className="flex justify-between items-center gap-2 px-3 w-[100%]">
                    <div className="text-2xl font-bold">Welcome to skillsync</div>
                    <AgentSheet />
                </div>
            </header>
            <div className="flex flex-col gap-8 h-[calc(100dvh-140px)] overflow-y-scroll">
                {chat?.map((res, index) => <> <div className="flex gap-5 justify-end" key={index}>
                    <div className="max-w-[60%] p-5">
                        {res.user}
                    </div>
                    <div className=" flex justify-center items-center w-[50px] h-[50px] bg-muted rounded-full ">
                        US
                    </div>
                </div>
                    <div className="flex gap-5 justify-start">
                        <div className=" flex justify-center items-center w-[50px] h-[50px] bg-[#000] text-white rounded-full">
                            AI
                        </div>
                        <div className="w-[60%] bg-muted p-4">
                            {/* {!res.ai && !submitted && <Image src={dot} alt="" className="rounded-full" />} */}
                            {res.ai}
                            {/* {!submitted && typeof res.ai !== "string" && <div className="flex items-center gap-2">
                                <Button className="w-[200px]"
                                    onClick={handleAccept} disabled={submitted}
                                >
                                    {submitted ? " Done" : "Accept"}
                                </Button>
                            </div>} */}
                        </div>
                    </div>
                </>)}
            </div>
            <div className="flex justify-center mt-auto w-[100%] p-2">
                <div className="mt-auto grow">
                    {isLoading && submitted && <div className={`absolute bottom-[89px] bg-muted w-[95%] overflow-scroll max-h-[400px] flex justify-between   gap-5 p-2 rounded-t-5`}>
                        {expandMessage && <div className="flex flex-col gap-2">
                            {data.map((res, index) => (<div className="flex gap-2" key={index}>
                                <div className="flex justify-center items-center rounded-full w-[40px] h-[40px]">
                                    <Image src={agent} alt="" className="rounded-full min-w-[40px] min-h-[40px]" />
                                </div>
                                <div className="">
                                    <p className="text-gray-500 text-md">{res && Object.keys(res)}</p>
                                    <p className="text-gray-500 text-sm">{res && JSON.stringify(res)}</p>
                                </div>
                            </div>))}
                        </div>}
                        {!expandMessage && submitted && <div>
                            <div className="flex items-center gap-2">
                                <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
                                    <Image src={agent} alt="" className="rounded-full" />
                                </div>
                                <div className="">
                                    <p className="text-gray-500 text-md">{data?.length > 0 && Object.keys(data[data.length - 1])}</p>
                                    <p className="text-gray-500 text-sm">{data?.length > 0 && JSON.stringify(data[data.length - 1])}</p>
                                </div>
                            </div>
                        </div>
                        }
                        <div>
                            <div className="flex gap-[10]"
                                onClick={() => setExpandMessage((prev) => !prev)}>
                                <AgentActivitySheet data={data} />
                                <div className="bg-white rounded-full p-2 cursor-pointer" >
                                    {expandMessage ? <ChevronDown /> : <ChevronUp />}
                                </div>
                            </div>
                        </div>
                    </div>
                    }
                    <div className="flex justify-center items-center gap-4 w-full">
                        <form onSubmit={handleSubmit} className="flex justify-center items-center gap-4 w-full">
                            <Input
                                className="h-[50px]"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                            />
                            <Button type="submit" size="sm" className="px-3">
                                Submit
                            </Button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    )
}
