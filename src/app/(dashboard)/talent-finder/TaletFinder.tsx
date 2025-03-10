'use client'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useState, useEffect } from "react"
import DialogS from "./dialog"
// import { Label } from "@/components/ui/label"
// import { useLottie } from "lottie-react";
// import Animation from "@/json/Animation.json";
import Image from "next/image"
import { useContext } from 'react';
import { UserContext } from "@/lib/providers";
import { useRouter } from "next/navigation"
import agent from "@/image/ai-agent.gif"
import github from "@/image/Github.gif"
import { ChevronUp, ChevronDown } from 'lucide-react';
import AgentActivitySheet from "./[slug]/AgentActivitySheet"
import { v4 as uuidv4 } from 'uuid';
import { Value } from "@radix-ui/react-select"
// import { SupervisorAgent } from "@/lib/agent"

export default function TaletFinder() {
    const [open, setOpen] = useState({ open: false, type: "" })
    const [loading, setLoading] = useState(false)
    const { items, setItems } = useContext(UserContext)
    const [ value, setValue ] = useState<string>("")
    const router = useRouter()
    const [JDcontent, setJDContent] = useState<string>("");

    // console.log(SupervisorAgent())

    const AiMagicIcon = () => (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width={48} height={48} color={"#cbc8c8"} fill={"none"}>
            <path d="M10 7L9.48415 8.39405C8.80774 10.222 8.46953 11.136 7.80278 11.8028C7.13603 12.4695 6.22204 12.8077 4.39405 13.4842L3 14L4.39405 14.5158C6.22204 15.1923 7.13603 15.5305 7.80278 16.1972C8.46953 16.864 8.80774 17.778 9.48415 19.6059L10 21L10.5158 19.6059C11.1923 17.778 11.5305 16.864 12.1972 16.1972C12.864 15.5305 13.778 15.1923 15.6059 14.5158L17 14L15.6059 13.4842C13.778 12.8077 12.864 12.4695 12.1972 11.8028C11.5305 11.136 11.1923 10.222 10.5158 8.39405L10 7Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
            <path d="M18 3L17.7789 3.59745C17.489 4.38087 17.3441 4.77259 17.0583 5.05833C16.7726 5.34408 16.3809 5.48903 15.5975 5.77892L15 6L15.5975 6.22108C16.3809 6.51097 16.7726 6.65592 17.0583 6.94167C17.3441 7.22741 17.489 7.61913 17.7789 8.40255L18 9L18.2211 8.40255C18.511 7.61913 18.6559 7.22741 18.9417 6.94166C19.2274 6.65592 19.6191 6.51097 20.4025 6.22108L21 6L20.4025 5.77892C19.6191 5.48903 19.2274 5.34408 18.9417 5.05833C18.6559 4.77259 18.511 4.38087 18.2211 3.59745L18 3Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
        </svg>
    );

    const [loadingIndex, setLoadingIndex] = useState(0);
    const [expandMessage, setExpandMessage] = useState(true);

    // const loadingMessages = [
    //   "Message sending to agent...",
    //   "Thinking...",
    //   "Recieve your message...",
    //   "Github agent invoked...",
    //   "Processing...",
    //   "Response generated...",
    //   "Finishing..."
    // ];

    const loadingMessages = [
        { message: "Message sending to agent...", agent: "agent" },
        { message: "Thinking...", agent: "agent" },
        { message: "Recieve your message...", agent: "agent" },
        { message: "Invoking github agent", agent: "agent" },
        { message: "Github agent invoke", agent: "github-agent" },
        { message: "Processing...", agent: "github-agent" },
        { message: "Response generated...", agent: "github-agent" },
        { message: "Finishing...", agent: "github-agent" },
        { message: "Final Answer generated", agent: "agent" },
        { message: "Finished...", agent: "agent" },

    ];

    const [displayedMessages, setDisplayedMessages] = useState<{ message: string; agent: string }[]>([]);


    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        // if (loading) {
        //     interval = setInterval(() => {
        //         setLoadingIndex((prev) => (prev + 1) % loadingMessages.length);
        //     }, 2000); // Change every 2 seconds
        // } else {
        //     setLoadingIndex(0); // Reset loading message when done
        // }

        if (loading) {
            interval = setInterval(() => {
                setLoadingIndex((prev) => {
                    if (prev < loadingMessages.length) {
                        setDisplayedMessages((msgs) => {
                            // Ensure the message is only added once
                            if (!msgs.includes(loadingMessages[prev])) {
                                return [...msgs, loadingMessages[prev]];
                            }
                            return msgs;
                        });
                        return prev + 1; // Increment without cycling
                    }
                    return prev;
                });
            }, 2000); // Change every 2 seconds
        } else {
            setLoadingIndex(0); // Reset loading message when done
            setDisplayedMessages([]); // Clear displayed messages
        }

        return () => clearInterval(interval);
    }, [loading]);

    const findMatch = () => {
        // setLoading(true)
        // new Promise<void>((resolve) => setTimeout(() => resolve(), 4000)).then(() => {
        // Copy the current items array (assuming "items" is coming from state)
        const updatedItems = [...items];

        // Find the index of the "Talent finder" item
        const finderIndex = updatedItems.findIndex(item => item.title === "Talent finder");
        if (finderIndex === -1) return; // Exit if not found

        // Clone the "Talent finder" object and its sub-items array
        const talentFinder = { ...updatedItems[finderIndex] };
        const subItems = [...(talentFinder.items || [])];

        // Find the highest number from the current "search X" titles
        let maxNumber = 0;
        subItems.forEach(subItem => {
            const match = subItem.title.match(/search (\d+)/);
            if (match) {
                const num = parseInt(match[1], 10);
                if (num > maxNumber) {
                    maxNumber = num;
                }
            }
        });

        // Create the new item with an incremented title and a dynamic URL
        const newItem = {
            title: `search ${maxNumber + 1}`,
            // url: `/talent-finder/${crypto.randomUUID()}`
            url: `/talent-finder/${uuidv4()}`
        };

        // Add the new item to the sub-items array
        subItems.push(newItem);

        // Update the "Talent finder" object with the new sub-items array
        talentFinder.items = subItems;
        updatedItems[finderIndex] = talentFinder;

        // Set the new items state
        setItems(updatedItems);

        // setLoading(false)
        // Simulate AI response delay
        // setTimeout(() => {

        //     setLoading(false); // Stop loading
        //     router.push(newItem?.url);
        // }, 16000); // Simulated delay (6 seconds)

        // console.log(value)
        router.push(`${newItem?.url}/?query=${value}`);
        // });
    }

    return (
        <>
            {open.open && <DialogS open={open} setOpen={setOpen} findMatch={findMatch} setJDContent={setValue} />}
            <div className={`flex flex-col justify-center items-center h-[88dvh] gap-5`}>
                <div className='flex flex-col justify-center items-center'>
                    <div className="text-4xl ">Skillsync by datacurve</div>
                    <div className="py-2 text-sm text-gray-500">Find your matching candidate with skillsync AI</div>
                    <div className="flex flex-col items-center">
                        <Button className="ml-auto" variant={"outline"} onClick={() => setOpen({ open: true, type: "github" })} >Upload job discription</Button>
                        <Input value={value} onChange={(value) => setValue(value.target.value)} className="w-[700px] h-[60px]" width={300} type="email" id="email" placeholder="Find your matching candidate with AI" />
                        {/* {loading && <div className="bg-muted w-full">
                            <div className={`bg-muted absolute flex justify-between w-[700px] gap-5 p-2 rounded-t-5`}>
                                {expandMessage && <div className="flex flex-col gap-2">
                                    <div className="flex items-center gap-2">
                                        <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
                                            <Image src={agent} alt="" className="rounded-full" />
                                        </div>
                                        <div className="">
                                            <p className="text-gray-500 text-md">Supervisor agent</p>
                                            <p className="text-gray-500 text-sm">{loadingMessages[loadingIndex]?.agent === "agent" && loadingMessages[loadingIndex]?.message}</p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
                                            <Image src={github} alt="" className="rounded-full" />
                                        </div>
                                        <div className="">
                                            <p className="text-gray-500 text-md">Github</p>
                                            <p className="text-gray-500 text-sm">{loadingMessages[loadingIndex]?.agent === "github-agent" && loadingMessages[loadingIndex]?.message}</p>
                                        </div>
                                    </div>
                                </div>}
                                {!expandMessage && <div>
                                    {loadingMessages[loadingIndex]?.agent === "agent" && <div className="flex items-center gap-2">
                                        <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
                                            <Image src={agent} alt="" className="rounded-full" />
                                        </div>
                                        <div className="">
                                            <p className="text-gray-500 text-md">Supervisor agent</p>
                                            <p className="text-gray-500 text-sm">{loadingMessages[loadingIndex]?.agent === "agent" && loadingMessages[loadingIndex]?.message}</p>
                                        </div>
                                    </div>}
                                    {loadingMessages[loadingIndex]?.agent === "github-agent" && <div className="flex items-center gap-2">
                                        <div className=" flex justify-center items-center w-[40px] h-[40px] text-white rounded-full">
                                            <Image src={github} alt="" className="rounded-full" />
                                        </div>
                                        <div className="">
                                            <p className="text-gray-500 text-md">Github</p>
                                            <p className="text-gray-500 text-sm">{loadingMessages[loadingIndex]?.agent === "github-agent" && loadingMessages[loadingIndex]?.message}</p>
                                        </div>
                                    </div>}
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
                        </div>} */}
                        <Button className="mt-5 cursor-pointer" size="lg" onClick={findMatch}>
                        {/* <Button className="mt-5 cursor-pointer" size="lg" onClick={SupervisorAgent}> */}
                            Find your match with skillsync AI
                            <AiMagicIcon />
                        </Button>
                        {/* {loading && <Image src={loader} alt="" />} */}
                    </div>
                </div>
            </div>
        </>
    )
}
