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
import { usePathname } from "next/navigation"
import { useRouter } from "next/navigation"
import ProfileCheckerDialog from "./profileCheckerDialog"
import ContentSections from "./ContentSections"
import ResumeSidebar from "./ResumeSidebar"

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
    const path = usePathname()
    const router = useRouter()
    const searchParams = useSearchParams()
    const search = searchParams.get('query')
    const scrollRef = useRef<HTMLDivElement>(null);
    const [open, setOpen] = useState({ open: false, type: "" })
    const [resumeUpload, setResumeUpload] = useState(false)

    const [selectedOptions, setSelectedOptions] = useState<string>("");
    const [submitted, setSubmitted] = useState(false);
    const [inputValue, setInputValue] = useState("")
    const [isLoading, setIsLoading] = useState(false);
    const [expandMessage, setExpandMessage] = useState(true);
    const [data, setData] = useState<StreamEvent[]>([]);

    const handleCheckboxChange = (value: string) => {
        console.log("value:", value)
        setSelectedOptions(value);
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
                        {/* {sessionStorage.getItem("selectedOptions") &&
                            JSON.parse(sessionStorage.getItem("selectedOptions") || "null").map(option =>
                                <div className="flex items-center space-x-2">
                                    <Checkbox
                                        id="github"
                                        // checked={selectedOptions}
                                        onCheckedChange={() => handleCheckboxChange(option)}
                                    />
                                    <Label htmlFor={option}>{option}</Label>
                                </div>)} */}
                        {sessionStorage.getItem("selectedOptions") &&
                            JSON.parse(sessionStorage.getItem("selectedOptions") || "null").includes("Github") &&
                            <div className="flex items-center space-x-2">
                                <Checkbox
                                    id="Github"
                                    checked={selectedOptions === "Github"}
                                    onCheckedChange={() => handleCheckboxChange("Github")}
                                />
                                <Label htmlFor="Github">Github</Label>
                            </div>}
                        {sessionStorage.getItem("selectedOptions") &&
                            JSON.parse(sessionStorage.getItem("selectedOptions") || "null").includes("Boolean agent") &&
                            <div className="flex items-center space-x-2">
                                <Checkbox
                                    id="Boolean agent"
                                    checked={selectedOptions === "Boolean agent"}
                                    onCheckedChange={() => handleCheckboxChange("Boolean agent")}
                                />
                                <Label htmlFor="Boolean agent">Boolean</Label>
                            </div>}
                        {sessionStorage.getItem("selectedOptions") &&
                            JSON.parse(sessionStorage.getItem("selectedOptions") || "null").includes("Resume match") &&
                            <div className="flex items-center space-x-2">
                                <Checkbox
                                    id="Resume match"
                                    checked={selectedOptions === "Resume match"}
                                    onCheckedChange={() => handleCheckboxChange("Resume match")}
                                />
                                <Label htmlFor="Resume match">Resume Match</Label>
                            </div>}
                        {sessionStorage.getItem("selectedOptions") &&
                            JSON.parse(sessionStorage.getItem("selectedOptions") || "null").includes("ATS") &&
                            <div className="flex items-center space-x-2">
                                <Checkbox
                                    id="ATS"
                                    checked={selectedOptions === "ATS"}
                                    onCheckedChange={() => handleCheckboxChange("ATS")}
                                />
                                <Label htmlFor="ATS">ATS</Label>
                            </div>}
                        {sessionStorage.getItem("selectedOptions") &&
                            JSON.parse(sessionStorage.getItem("selectedOptions") || "null").includes("Reddit") &&
                            <div className="flex items-center space-x-2">
                                <Checkbox
                                    id="Reddit"
                                    checked={selectedOptions === "Reddit"}
                                    onCheckedChange={() => handleCheckboxChange("Reddit")}
                                />
                                <Label htmlFor="Reddit">Reddit</Label>
                            </div>}
                        {/* <div className="flex items-center space-x-2">
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
                        </div> */}

                        {/* {submitted && getSelectedItems().length > 0 && (
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
                        )} */}
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


    // const messages = [
    //     {
    //         id: "12121",
    //         user: (search ? search.replace("+", " ") : ""),
    //         ai: <SelectPlatform />
    //     },
    // ]
    const [chat, setChat] = useState<{ id: string; user: string; ai: string | React.ReactNode }[]>([])

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
        const questioWithTeam = `${props.user} "Team: ${selectedOptions}"`

        console.log("questioWithTeam", questioWithTeam)

        const data = await streamFn(questioWithTeam)

        const secondlastNode = data[data.length - 2]

        const nodeName = Object.keys(secondlastNode)[0]
        // console.log("nodeName", nodeName)
        const lastContent = secondlastNode[nodeName]?.messages[0]?.kwargs?.content

        const response = await sendMessage(lastContent);

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
    console.log("chat is loaded", chat)
    const handleAccept = async () => {
        router.push(`${path}?query=${search}&platform="${selectedOptions}"`)
        setSubmitted(true);
        if (selectedOptions !== "Resume match") {

            setIsLoading(true);
            const newMessage = {
                id: uuidv4(),
                user: (search ? search.replace("+", " ") : ""),
                ai: <div className="loader" />, // Placeholder for AI response
            }

            console.log("selectedOptions", selectedOptions)
            setChat((prevMessages) => [...prevMessages, newMessage]);
            await handleUpdateMessageStream(newMessage)
            setExpandMessage(false)
        }
    };



    const addMessage = async (userMessage: string) => {
        setIsLoading(true); // Start loading when user sends message

        const id = uuidv4()

        const newMessage = {
            id,
            user: userMessage,
            ai: <div className="loader" />
        };
        setChat((prevChat) => [...prevChat, newMessage]);

        await handleUpdateMessageStream(newMessage)
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
                {!submitted && <SelectPlatform />}
                {submitted && chat?.map((res, index) => <> <div className="flex gap-5 justify-end" key={index}>
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
                            {res.ai}
                        </div>
                    </div>
                </>)}
                {submitted && selectedOptions === "Resume match" && !resumeUpload && <div className="flex justify-center items-center gap-4 w-full h-[100%]">
                    <div>
                        <Button onClick={() => setOpen({ open: true, type: "resume" })}>
                            Click to start your profile matches.
                        </Button>
                    </div>
                </div>}
                {submitted && selectedOptions === "Resume match" && resumeUpload && <div className="flex gap-4 w-full h-[100%]">
                    <ContentSections />
                    <div className="relative right-0">
                        <ResumeSidebar />
                    </div>
                </div>}
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
                    {submitted && selectedOptions !== "Resume match" && <div className="flex justify-center items-center gap-4 w-full">
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
                    </div>}
                </div>
            </div>
            {open.open && <ProfileCheckerDialog open={open} setOpen={setOpen} setResumeUpload={setResumeUpload} />}
        </div>
    )
}
